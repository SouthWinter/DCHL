"""Construct an optional action mask by deleting replay-redundant actions.

The original training history stays intact. Only the supervision mask changes.
This is a privileged-state local labeling procedure, not HSL's LLM labeler.
"""
import argparse
import json
from pathlib import Path
import re
import time


class CachedReplay:
    """Reuse one compiled PDDL task and repeated action sequences per rollout.

    Native applicability and state transitions are unchanged. Rendering is only
    needed for admissible commands, not for feedback text in this label search.
    The normal environment remains the source of the training observations.
    """
    def __init__(self, env, steps):
        from alfworld_hindsight import facts_of
        state = env.reset()
        self.core = env.unwrapped
        self.pddl = self.core._pddl_state
        self.initial_facts = tuple(self.pddl.facts)
        self.initial_state = state
        self.initial_operators = dict(self.pddl._operators)
        self.actions = [step['action'] for step in steps]
        self.cache = {(): facts_of(state)}
        self.executed = self.hits = 0

    def reset(self):
        from collections import Counter, defaultdict
        self.pddl.downward_lib.load_sas(self.pddl.sas.encode('utf-8'))
        # Match PddlState's fact indexes; its GameLogic is distinct from the
        # generic textworld.logic.GameLogic accepted by State.__init__.
        self.pddl._facts = defaultdict(set)
        self.pddl._vars_by_name = {}
        self.pddl._vars_by_type = defaultdict(set)
        self.pddl._var_counts = Counter()
        self.pddl.add_facts(self.initial_facts)
        self.pddl._operators = dict(self.initial_operators)
        self.core._pddl_state = self.pddl
        self.core.state = self.initial_state
        return self.initial_state

    def __call__(self, indices):
        from alfworld_hindsight import facts_of
        from textworld.core import GameState
        key = tuple(self.actions[i] for i in indices)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        state = self.reset()
        for command in key:
            if command not in state['_valid_commands']:
                self.cache[key] = None
                return None
            index = state['_valid_commands'].index(command)
            action = state['_valid_actions'][index]
            self.pddl.apply(action)
            self.executed += 1
            self.core._last_action = action
            self.core.state = GameState()
            self.core._gather_infos()
            state = self.core.state
        result = facts_of(state)
        self.cache[key] = result
        return result


def reduce_actions(length, protected, executable_goal, removal_blocks=()):
    """Greedy block deletion; returns a verified subset, not a global optimum."""
    keep = list(range(length))
    checks = 0
    if not executable_goal(keep):
        raise ValueError('original prefix does not satisfy the replay target')
    # Moving an unrelated object can require several mutually dependent actions.
    # Try complete excursions returning to the same location before generic
    # block deletion, which may split such a cycle at an unlucky boundary.
    for block in removal_blocks:
        remove = set(block) - protected
        candidate = [i for i in keep if i not in remove]
        if candidate == keep:
            continue
        checks += 1
        if executable_goal(candidate):
            keep = candidate
    size = 1 << max(0, (length.bit_length() - 1))
    while size:
        changed = True
        while changed:
            changed = False
            for start in range(0, len(keep), size):
                remove = set(keep[start:start + size]) - protected
                if not remove:
                    continue
                candidate = [i for i in keep if i not in remove]
                checks += 1
                if executable_goal(candidate):
                    keep = candidate
                    changed = True
                    break
        size //= 2
    return keep, checks


def protected_steps(prefix, goal, preserve_exploration=True):
    queries = {i for i, step in enumerate(prefix) if
        step['action'] in ('look', 'inventory') or step['action'].startswith('examine ')}
    exploration = set()
    if preserve_exploration:
        # Keep first visits/openings until all selected goal objects have appeared
        # in real observations. This conservative rule also keeps unsuccessful
        # search locations; hidden-state replay alone would discard that evidence.
        objects = {f[1] for f in goal['witness_facts'] if f[0] == 'objecttype'}
        discovery = []
        for obj in objects:
            pattern = re.compile(r'(?<!\w)' + re.escape(obj) + r'(?!\w)', re.I)
            first = len(prefix) - 1
            for i, step in enumerate(prefix):
                if pattern.search(step['observation']):
                    first = i - 1
                    break
                if pattern.search(step['result_observation']):
                    first = i
                    break
            discovery.append(first)
        boundary = max(discovery, default=-1)
        seen = set()
        for i, step in enumerate(prefix):
            action = step['action']
            if i <= boundary and action.startswith(('go to ', 'open ')) and action not in seen:
                exploration.add(i)
                seen.add(action)
    return queries, exploration


def label_goal(env, steps, goal, preserve_exploration=True, replay_cache=None):
    from alfworld_hindsight import facts_of
    prefix = steps[:goal['prefix_length']]
    target = {tuple(f) for f in goal['witness_facts']}
    queries, exploration = protected_steps(prefix, goal, preserve_exploration)
    protected = queries | exploration
    executed = 0

    def replay(indices):
        nonlocal executed
        if replay_cache is not None:
            before = replay_cache.executed
            facts = replay_cache(indices)
            executed += replay_cache.executed - before
            return facts is not None and target.issubset(facts)
        state = env.reset()
        for i in indices:
            action = prefix[i]['action']
            if action not in state.admissible_commands:
                return False
            state, _, _ = env.step(action)
            executed += 1
        return target.issubset(facts_of(state))

    started = time.monotonic()
    visits = {}
    blocks = []
    for i, step in enumerate(prefix):
        action = step['action']
        if action.startswith('go to '):
            for previous in visits.get(action, []):
                blocks.append(range(previous + 1, i + 1))
            visits.setdefault(action, []).append(i)
    blocks.sort(key=lambda block: (-len(block), block.start))
    keep, checks = reduce_actions(len(prefix), protected, replay, blocks if preserve_exploration else ())
    return dict(action_loss_mask=[i in keep for i in range(len(prefix))],
        kept_indices=keep, relevant_actions=len(keep), original_actions=len(prefix),
        protected_query_indices=sorted(queries), protected_exploration_indices=sorted(exploration),
        replay_checks=checks + 1,
        replayed_actions=executed, wall_seconds=time.monotonic() - started,
        replay_backend='cached_compiled_pddl' if replay_cache is not None else 'full_environment',
        rule=('greedy_goal_replay_with_observed_search_v3' if preserve_exploration
              else 'greedy_executable_goal_preserving_deletion_v1'),
        limitation='Order-dependent deletion under privileged state; retained search is a heuristic, not minimal information sufficiency; original observations remain training inputs.')


def main():
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=Path, required=True)
    parser.add_argument('--rollout-id', type=int, required=True)
    parser.add_argument('--goal-index', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state-only', action='store_true', help='Reproduce the earlier state-only diagnostic.')
    args = parser.parse_args()
    rows = (json.loads(line) for line in args.rollouts.read_text().splitlines())
    row = next(row for row in rows if row['rollout_id'] == args.rollout_id)
    h = row['hindsight']; goal = h['goals'][args.goal_index]
    env = textworld.start(row['gamefile'], request_infos=textworld.EnvInfos(
        facts=True, won=True, admissible_commands=True), wrappers=[AlfredDemangler(shuffle=False)])
    try:
        result = label_goal(env, h['steps'], goal, preserve_exploration=not args.state_only)
    finally:
        env.close()
    result.update(source=str(args.rollouts), rollout_id=args.rollout_id,
        goal_index=args.goal_index, instruction=goal['instruction'],
        original_mask=goal['action_loss_mask'],
        retained_actions=[h['steps'][i]['action'] for i in result['kept_indices']],
        training_history_changed=False, used_for_training=False)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
