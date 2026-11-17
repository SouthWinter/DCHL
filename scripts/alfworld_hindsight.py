"""Environment-state hindsight goals, shared by both local HSL comparison arms.

This is a privileged-state adaptation, not HSL's 70B text relabeler. It never
requests a planner or reads an expert walkthrough. Output contains actual
executed prefixes; action relevance and training weights are separate stages.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from itertools import product
from pathlib import Path
import re
import time

import collect_alfworld_takeover_frontier as core

PLACE = 'pick_and_place_simple'
TWO = 'pick_two_obj_and_place'
EXAMINE = 'look_at_obj_in_light'
PROPERTIES = {
    'isclean': ('cleanable', 'pick_clean_then_place_in_recep', 'clean'),
    'ishot': ('heatable', 'pick_heat_then_place_in_recep', 'heat'),
    'iscool': ('coolable', 'pick_cool_then_place_in_recep', 'cool'),
}


def goal_instruction(family, obj, destination):
    """Shared text rendering for hindsight goals and declared original goals."""
    if family == PLACE:
        return f'put a {obj} in/on the {destination}.'
    if family == TWO:
        return f'put two {obj} in/on the {destination}.'
    if family == EXAMINE:
        return f'examine the {obj} with the {destination}.'
    for _, (_, kind, verb) in PROPERTIES.items():
        if family == kind:
            return f'{verb} a {obj} and put it in/on the {destination}.'
    raise ValueError(f'Unsupported goal family: {family}')


def facts_of(state):
    return {(p.name.lower(), *(a.name.strip().lower() for a in p.arguments))
            for p in state.facts}


def type_name(value):
    return value.removesuffix('type')


def negate(fact):
    return ((fact[0][4:] if fact[0].startswith('not_') else 'not_' + fact[0]), *fact[1:])


def action_preconditions(env, state, command):
    """Ground a satisfied normalized PDDL branch of the actual command.

    TextWorld's public Action.preconditions are empty for this backend. Its
    normalized domain can contain several branches with the same action name;
    choosing the last branch alone incorrectly requires opening a dresser.
    """
    index = state['_valid_commands'].index(command)
    action = state['_valid_actions'][index]
    binding = {'?' + k.name: v.name for k, v in action.mapping.items()}
    raw = {(p.name, *(v.name for v in p.arguments)) for p in state['_facts']}

    def evaluate(node):
        if hasattr(node, 'predicate'):
            args = tuple(binding.get(v, v) for v in node.args)
            if any(v.startswith('?') for v in args):
                raise ValueError('unbound normalized PDDL parameter')
            if node.predicate == '=':
                ok = args[0] == args[1]
                return set() if ok != node.negated else None
            fact = (node.predicate, *args)
            if (fact in raw) == node.negated:
                return None
            return {negate(fact) if node.negated else fact}
        if type(node).__name__ == 'Conjunction':
            found = [evaluate(p) for p in node.parts]
            return None if any(x is None for x in found) else set().union(*found)
        if type(node).__name__ == 'Disjunction':
            found = [x for p in node.parts if (x := evaluate(p)) is not None]
            return min(found, key=lambda x: (len(x), sorted(x))) if found else None
        if type(node).__name__ == 'Truth':
            return set()
        raise ValueError(f'unsupported normalized condition: {type(node).__name__}')

    candidates = []
    task = env._pddl_state.task
    parents = {t.name: t.basetype_name for t in task.types}

    def has_type(obj, required):
        current, visited = obj.type_name, set()
        while current is not None and current not in visited:
            if current == required:
                return True
            visited.add(current)
            current = parents.get(current)
        return False

    original_binding = dict(binding)
    for domain_action in task.actions:
        if domain_action.name != action.name:
            continue
        # The executable command omits existential witnesses: "examine bed"
        # binds agent/receptacle but its normalized precondition also has a
        # location. Find a typed witness satisfying the actual current facts.
        missing = [p for p in domain_action.parameters if p.name not in original_binding]
        choices = [[o.name for o in task.objects if has_type(o, p.type_name)] for p in missing]
        for values in product(*choices):
            binding = dict(original_binding, **dict(zip((p.name for p in missing), values)))
            found = evaluate(domain_action.precondition)
            if found is not None:
                candidates.append(found)
    if not candidates:
        raise ValueError(f'no satisfied PDDL branch for {command}')
    selected = min(candidates, key=lambda x: (len(x), sorted(x)))
    infos = state['_entity_infos']
    return {(f[0], *(infos[v].name.strip().lower() for v in f[1:])) for f in selected}


def relevance_mask(steps, goal):
    """Backward state support plus explicitly retained information queries.

    This is an executable-state relevance rule shared by both arms, not the
    paper's LLM relevance labels and not a minimal causal explanation. Queries
    remain supervised because state-change-only rules miss information value.
    Every masked action still remains in later model inputs.
    """
    prefix = steps[:goal['prefix_length']]
    needed = {tuple(f) for f in goal['witness_facts']}
    mask = [False] * len(prefix)
    reasons = ['unrelated_state_effect'] * len(prefix)
    for i in range(len(prefix) - 1, -1, -1):
        step = prefix[i]
        adds = {tuple(f) for f in step['added_facts']}
        adds |= {negate(tuple(f)) for f in step['removed_facts']}
        query = step['action'] in ('look', 'inventory') or step['action'].startswith('examine ')
        if adds & needed or query:
            mask[i] = True
            reasons[i] = 'information_query' if query else 'goal_state_support'
            needed.difference_update(adds)
            needed.update(tuple(f) for f in step['precondition_facts'])
    return mask, reasons


def achieved_goals(facts):
    """Match the six released ALFWorld PDDL goal families.

    Return one concrete witness per goal (family, object type, destination type).
    The two-object predicate requires distinct instances in the SAME receptacle.
    Examination follows released PDDL: toggled lamp, co-location and held object.
    PDDL uses istoggled, not ison; these have different meanings.
    """
    by = defaultdict(list)
    for fact in sorted(facts):
        by[fact[0]].append(fact[1:])
    objects = dict(by['objecttype'])
    receptacles = dict(by['receptacletype'])
    locations = dict(by['receptacleatlocation'])
    out = {}

    def add(family, obj_type, dest_type, instruction, witness):
        key = (family, type_name(obj_type), type_name(dest_type))
        out.setdefault(key, {'task_type': family, 'object_type': key[1],
            'destination_type': key[2], 'instruction': instruction,
            'witness_facts': [list(f) for f in sorted(witness)]})

    groups = defaultdict(list)
    for obj, rec in by['inreceptacle']:
        if obj not in objects or rec not in receptacles:
            continue
        ot, rt = objects[obj], receptacles[rec]
        on, rn = type_name(ot), type_name(rt)
        witness = {('inreceptacle', obj, rec), ('objecttype', obj, ot),
                   ('receptacletype', rec, rt)}
        add(PLACE, ot, rt, goal_instruction(PLACE, on, rn), witness)
        groups[(ot, rec)].append(obj)
        for prop, (capability, family, verb) in PROPERTIES.items():
            if (prop, obj) in facts and (capability, obj) in facts:
                add(family, ot, rt, goal_instruction(family, on, rn),
                    witness | {(prop, obj), (capability, obj)})
    for (ot, rec), objs in sorted(groups.items()):
        if len(set(objs)) >= 2:
            rt = receptacles[rec]
            witness = {('receptacletype', rec, rt)}
            for obj in sorted(set(objs))[:2]:
                witness |= {('objecttype', obj, ot), ('inreceptacle', obj, rec)}
            add(TWO, ot, rt, goal_instruction(TWO, type_name(ot), type_name(rt)), witness)
    for lamp, rec in by['inreceptacle']:
        lt = objects.get(lamp, '')
        if lt not in ('desklamptype', 'floorlamptype') or rec not in locations:
            continue
        if ('toggleable', lamp) not in facts or ('istoggled', lamp) not in facts:
            continue
        for agent, loc in by['atlocation']:
            if loc != locations[rec]:
                continue
            for holder, obj in by['holds']:
                if holder == agent and obj in objects:
                    ot = objects[obj]
                    witness = {('objecttype', lamp, lt), ('toggleable', lamp),
                        ('istoggled', lamp), ('receptacleatlocation', rec, loc),
                        ('atlocation', agent, loc), ('inreceptacle', lamp, rec),
                        ('objecttype', obj, ot), ('holds', agent, obj)}
                    add(EXAMINE, ot, lt,
                        goal_instruction(EXAMINE, type_name(ot), type_name(lt)), witness)
    return out


def replace_instruction(observation, instruction):
    # Only the task instruction changes; environmental observations stay intact.
    return re.sub(r'Your task is to: [^\n]*',
                  'Your task is to: ' + instruction, observation)


def prefix_messages(prefix, instruction):
    """Rebuild the existing four-history interface under the relabeled goal.

    No action is removed. Relevance masks must be applied by the next stage;
    this function deliberately does not produce train-ready loss masks.
    """
    history, records = [], []
    for step in prefix:
        observation = replace_instruction(step['observation'], instruction)
        records.append({'messages': [
            {'role': 'system', 'content': core.SYSTEM_PROMPT},
            {'role': 'user', 'content': core.prompt(instruction, observation,
                step['admissible_actions'], history)},
            {'role': 'assistant', 'content': f"<action>{step['action']}</action>"}],
            'step': len(history), 'action': step['action']})
        history.append({'observation': observation, 'action': step['action']})
    return records


def replay_and_relabel(row, relevance_rule='backward'):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    if relevance_rule not in ('backward', 'replay_search'):
        raise ValueError(f'unknown relevance rule: {relevance_rule}')

    started = time.monotonic()
    env = textworld.start(row['gamefile'],
        request_infos=textworld.EnvInfos(facts=True, won=True, admissible_commands=True),
        wrappers=[AlfredDemangler(shuffle=False)])
    try:
        state = env.reset()
        initial = facts_of(state)
        seen = set(achieved_goals(initial))
        prefixes, steps = [], []
        for index, source in enumerate(row['trajectory']):
            action = source.get('action', source.get('student_action'))
            observation = str(state.feedback)
            if observation.strip() != source['observation'].strip():
                raise ValueError(f'replay observation differs before step {index}')
            executable=action in state.admissible_commands
            if not executable and not row.get('dialogue_hsl',False):
                raise ValueError(f'non-executable action at step {index}: {action}')
            before = facts_of(state)
            admissible = list(state.admissible_commands)
            preconditions = action_preconditions(env, state, action) if executable else set()
            state, _, done = env.step(action)
            if str(state.feedback).strip() != source['result_observation'].strip():
                raise ValueError(f'replay observation differs after step {index}')
            after = facts_of(state)
            steps.append({'step': index, 'observation': observation, 'action': action,
                'admissible_actions': admissible, 'result_observation': str(state.feedback),
                'precondition_facts': [list(f) for f in sorted(preconditions)],
                'added_facts': [list(f) for f in sorted(after - before)],
                'removed_facts': [list(f) for f in sorted(before - after)]})
            for key, goal in achieved_goals(after).items():
                if key in seen:
                    continue
                seen.add(key)
                labeled_goal = dict(goal, first_achievement_step=index + 1,
                    # Reference a shared executed history instead of copying every prefix.
                    prefix_length=index + 1)
                mask, reasons = relevance_mask(steps, labeled_goal)
                if row.get('dialogue_hsl',False):
                    for j,s in enumerate(steps):
                        if s['action'] not in s['admissible_actions']:
                            mask[j]=False;reasons[j]='invalid_generated_action'
                n, length = sum(mask), len(mask)
                if n == 0:
                    raise ValueError('newly achieved goal has no supporting action')
                labeled_goal.update(action_loss_mask=mask, relevance_reasons=reasons,
                    relevant_actions=n, hsl_weight=(n / length) ** 0.8 * n,
                    relevance_rule='pddl_backward_support_plus_queries_v1')
                prefixes.append(labeled_goal)
            if done or state.won:
                if index + 1 != len(row['trajectory']):
                    raise ValueError('source continues after environment termination')
                break
        if bool(state.won) != bool(row['won']):
            raise ValueError('replayed terminal success differs from source')
        relevance_seconds = 0.0
        if relevance_rule == 'replay_search':
            from alfworld_replay_relevance import CachedReplay, label_goal
            relevance_started = time.monotonic()
            replay_cache = CachedReplay(env, steps) if prefixes else None
            for goal in prefixes:
                revised = label_goal(env, steps, goal, replay_cache=replay_cache)
                goal['backward_action_loss_mask'] = goal['action_loss_mask']
                goal['backward_hsl_weight'] = goal['hsl_weight']
                n, length = revised['relevant_actions'], goal['prefix_length']
                goal.update(action_loss_mask=revised['action_loss_mask'],
                    relevant_actions=n, hsl_weight=(n / length) ** 0.8 * n,
                    relevance_rule=revised['rule'], replay_relevance=revised,
                    relevance_reasons=[('retained_by_replay_or_search' if keep else 'replay_redundant')
                                       for keep in revised['action_loss_mask']])
            relevance_seconds = time.monotonic() - relevance_started
        return {'gamefile': row['gamefile'], 'requested_task_type': row['task_type'],
            'rollout_id': row.get('rollout_id'), 'policy_version': row.get('policy_version'),
            'source_success': bool(state.won), 'steps': steps, 'goals': prefixes,
            'initial_goals_excluded': len(achieved_goals(initial)),
            'relabeler': 'alfworld_pddl_state_v1', 'privileged_state': True,
            'action_relevance_ready': True, 'training_ready': False,
            'relevance_mode': relevance_rule, 'relevance_worker_seconds': relevance_seconds,
            'wall_seconds': time.monotonic() - started}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engineering-fixtures', action='store_true')
    parser.add_argument('--max-per-class', type=int, default=0)
    parser.add_argument('--outcome', choices=['all', 'success', 'failure'], default='all')
    parser.add_argument('--relevance-rule', choices=['backward', 'replay_search'], default='backward')
    args = parser.parse_args()
    counts, produced, total_goals = Counter(), Counter(), Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as source, args.output.open('x') as output:
        for line in source:
            row = json.loads(line)
            if row.get('status') != 'complete':
                continue
            if args.outcome != 'all' and bool(row['won']) != (args.outcome == 'success'):
                continue
            family = row['task_type']
            if args.max_per_class and counts[family] >= args.max_per_class:
                continue
            if not args.engineering_fixtures and not row.get('policy_version'):
                raise ValueError('formal rollouts require policy_version')
            result = replay_and_relabel(row, args.relevance_rule)
            result['engineering_fixture_only'] = args.engineering_fixtures
            output.write(json.dumps(result) + '\n')
            output.flush()
            counts[family] += 1
            produced.update((family, g) for g in {x['task_type'] for x in result['goals']})
            total_goals.update(g['task_type'] for g in result['goals'])
            print(json.dumps({'processed': sum(counts.values()),
                'goals': len(result['goals']), 'seconds': result['wall_seconds']}), flush=True)
    summary = {'attempts_by_requested_class': dict(counts),
        'goal_counts': dict(total_goals), 'production_probabilities': [
            {'requested': a, 'produced': b, 'positive_attempts': n,
             'attempts': counts[a], 'probability': n / counts[a]}
            for (a, b), n in sorted(produced.items())],
        'engineering_fixture_only': args.engineering_fixtures,
        'source_outcome_filter': args.outcome,
        'training_ready': False}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
