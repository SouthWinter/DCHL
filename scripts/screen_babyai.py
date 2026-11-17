"""TRAIN-only BabyAI feasibility screen; native verifiers and public macro actions.

Run with external/babyai_runtime/bin/python. No training or TEST execution.
Deletion supports are one replay-valid subset, not a unique causal explanation.
"""
import argparse
import collections
import copy
import json
import os
from pathlib import Path
import re
import sys
import time
import warnings

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
warnings.filterwarnings('ignore', category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'external/babyai_native_v1'))
import minigrid  # registers the official task IDs
from environment import BabyAI, all_levels
from minigrid.envs.babyai.core import verifier as V

DATA = ROOT / 'data/babyai_screen_v1'
OUT = ROOT / 'analysis/babyai_screen_v1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def encode(instr):
    result = {'kind': type(instr).__name__}
    for k in ['desc', 'desc_move', 'desc_fixed']:
        if hasattr(instr, k):
            d = getattr(instr, k)
            result[k] = dict(type=d.type, color=d.color, loc=d.loc)
    for k in ['instr_a', 'instr_b']:
        if hasattr(instr, k):
            result[k] = encode(getattr(instr, k))
    if hasattr(instr, 'strict'):
        result['strict'] = instr.strict
    return result


def decode(spec):
    cls = getattr(V, spec['kind'])
    kw = {k: V.ObjDesc(**spec[k]) for k in ['desc', 'desc_move', 'desc_fixed'] if k in spec}
    if 'instr_a' in spec:
        return cls(decode(spec['instr_a']), decode(spec['instr_b']), strict=spec.get('strict', False))
    args = list(kw.values())
    return cls(*args, **({'strict': spec['strict']} if 'strict' in spec else {}))


def environment(item):
    idx = int(item.split('_')[-1])
    return BabyAI(game_name=all_levels[idx % 40 + 1], seed=idx // 40)


def candidates(env):
    root = encode(env.env.unwrapped.instrs)
    specs = []
    def add_tree(s):
        specs.append(s)
        for k in ['instr_a', 'instr_b']:
            if k in s:
                add_tree(s[k])
    add_tree(root)
    cells = env.env.unwrapped.grid.grid
    for typ, color in sorted({(o.type, o.color) for o in cells if o is not None and o.type in V.OBJ_TYPES}):
        for kind in ['GoToInstr', 'OpenInstr' if typ == 'door' else 'PickupInstr']:
            specs.append(dict(kind=kind, desc=dict(type=typ, color=color, loc=None)))
    unique = {}
    for s in specs:
        ins = decode(s)
        text = ins.surface(env.env.unwrapped)
        unique.setdefault(text, s)
    return [dict(instruction=k, spec=v) for k, v in unique.items()]


def replay(item, actions, specs=None, keep_trace=False, native_goal=None):
    env = environment(item)
    native = env.env.unwrapped
    if native_goal is not None:
        native.instrs = decode(native_goal)
        native.instrs.reset_verifier(native)
    object_ids = {id(native.grid.get(x, y)): [x, y]
                  for x in range(native.grid.width) for y in range(native.grid.height)
                  if native.grid.get(x, y) is not None}
    def object_identity(obj):
        return None if obj is None else [obj.type, obj.color, object_ids.get(id(obj))]
    specs = candidates(env) if specs is None else specs
    validators = []
    initially_satisfied = set()
    for i, goal in enumerate(specs):
        v = decode(goal['spec'])
        v.reset_verifier(native)
        if isinstance(v, V.GoToInstr) and v.verify(native.actions.done) == 'success':
            initially_satisfied.add(i)
        validators.append(v)
    hits = {}
    native_success_steps = []
    primitive_counts = []
    trajectory = []
    semantic_targets = []
    interaction_targets = []
    current = -1
    original_step = env.env.step
    def traced_step(action):
        if int(action) in [3, 4, 5]:
            interaction_targets.append(object_identity(native.carrying if int(action) == 4 else native.grid.get(*native.front_pos)))
        result = original_step(action)
        primitive_counts[current] += 1
        if result[1] > 0:
            native_success_steps.append(current)
        for i, verifier in enumerate(validators):
            if int(action) == int(native.actions.drop):
                native.update_objs_poss(verifier)
            if i not in hits and i not in initially_satisfied and verifier.verify(action) == 'success':
                hits[i] = current
        return result
    env.env.step = traced_step
    initial = env._get_obs()
    mission = native.mission
    prompt_mission = env.goal
    try:
        for current, action in enumerate(actions):
            interaction_targets.clear()
            primitive_counts.append(0)
            before = env._get_obs()
            available = action in env.action_space
            _, reward, done, _ = env.step(action)
            if action.startswith(('go to ', 'go through ')):
                target = object_identity(native.grid.get(*(native.agent_pos if action.startswith('go through ') else native.front_pos)))
            elif action.startswith(('pick', 'toggle', 'drop')):
                target = copy.deepcopy(interaction_targets)
            else:
                target = None
            semantic_targets.append(dict(available=available, target=target))
            if keep_trace:
                trajectory.append(dict(index=current, action=action, available=available,
                    observation_before=before, observation_after=env._get_obs(), reward=reward,
                    done=done, primitive_steps=primitive_counts[-1]))
        return dict(item_id=item, mission=mission, prompt_mission=prompt_mission,
                    initial_observation=initial, root_kind=type(native.instrs).__name__,
                    native_success=bool(native_success_steps), native_success_steps=native_success_steps,
                    wrapper_success=bool(env.done and env.reward > 0), wrapper_reward=env.reward,
                    actions=actions, primitive_counts=primitive_counts, hits=hits,
                    candidates=specs, trajectory=trajectory, semantic_targets=semantic_targets)
    finally:
        env.env.close()


def add_supports(result):
    item, actions, specs = result['item_id'], result['actions'], result['candidates']
    cache = {}
    def succeeds(indices):
        key = tuple(indices)
        if key not in cache:
            trial = replay(item, [actions[j] for j in indices], specs)
            # Ordinal IDs such as "ball 1" are recomputed after every action.
            # A shortened replay must still address the same physical objects.
            same_targets = all(a == result['semantic_targets'][j]
                               for j, a in zip(indices, trial['semantic_targets']))
            cache[key] = trial['hits'] if same_targets else {}
        return cache[key]
    labels = []
    for i, end in result['hits'].items():
        selected = list(range(end + 1))
        for j in reversed(range(end + 1)):
            trial = [k for k in selected if k != j]
            if i in succeeds(trial):
                selected = trial
        assert selected and i in succeeds(selected)
        # Independent check: actually replace the native task verifier and
        # require native positive reward, not just our parallel observer.
        certificate = replay(item, [actions[j] for j in selected], [], native_goal=specs[i]['spec'])
        assert certificate['native_success'], (item, specs[i], selected)
        labels.append(dict(**specs[i], prefix_length=end + 1, support=selected,
                           native_goal_reward_verified=True))
    supports = {tuple(g['support']) for g in labels}
    pairs = [(set(a), set(b)) for a in supports for b in supports if a < b]
    result['labels'] = labels
    result['support_semantics_version'] = 4
    result['support_summary'] = dict(labels=len(labels), unique_supports=len(supports),
        distinct_supports=len(supports) >= 2,
        overlap_and_each_unique=any(a & b and a - b and b - a for a, b in pairs),
        nested_different=any(a != b and (a <= b or b <= a) for a, b in pairs),
        each_unique=any(a - b and b - a for a, b in pairs),
        replay_queries=len(cache))
    result.pop('candidates')
    result.pop('hits')
    return result


def demonstration_actions(row):
    return [re.split(r'Action\s*:\s*', m['value'], flags=re.I)[-1].strip().lower()
            for m in row['conversations'] if m['from'] == 'gpt' and m.get('loss') is True]


def summarize(results):
    return dict(sources=len(results), native_success=sum(r['native_success'] for r in results),
        wrapper_success=sum(r['wrapper_success'] for r in results),
        native_wrapper_disagree=sum(r['native_success'] != r['wrapper_success'] for r in results),
        changed_order_prompts=sum(r['mission'] != r['prompt_mission'] for r in results),
        macro_actions=sum(len(r['actions']) for r in results),
        primitive_actions=sum(sum(r['primitive_counts']) for r in results),
        one_macro_sources=sum(len(r['actions']) <= 1 for r in results),
        goals=sum(r['support_summary']['labels'] for r in results),
        multi_goal_sources=sum(r['support_summary']['labels'] >= 2 for r in results),
        distinct_support_sources=sum(r['support_summary']['distinct_supports'] for r in results),
        overlap_each_unique_sources=sum(r['support_summary']['overlap_and_each_unique'] for r in results),
        each_unique_sources=sum(r['support_summary']['each_unique'] for r in results),
        nested_different_sources=sum(r['support_summary']['nested_different'] for r in results),
        instruction_types=dict(collections.Counter(r['root_kind'] for r in results)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--per-level', type=int, default=5)
    args = p.parse_args()
    started = time.monotonic()
    train = json.loads((DATA / 'source/babyai_train.json').read_text())
    test = json.loads((DATA / 'source/babyai_test.json').read_text())
    assert not {r['item_id'] for r in train} & {r['item_id'] for r in test}
    groups = collections.defaultdict(list)
    for row in train:
        groups[int(row['item_id'].split('_')[-1]) % 40 + 1].append(row)
    selected = [row for level in sorted(groups)
                for row in sorted(groups[level], key=lambda x: int(x['item_id'].split('_')[-1]))[:args.per_level]]
    save(DATA / 'screen_manifest.json', dict(train_count=len(train), test_count=len(test),
        train_test_overlap=0, selection='first five TRAIN world seeds per published level; frozen before replay',
        selected_ids=[r['item_id'] for r in selected], test_execution=0,
        levels={str(k): dict(name=all_levels[k], train=len(v), test=sum(int(r['item_id'].split('_')[-1]) % 40 + 1 == k for r in test)) for k, v in sorted(groups.items())},
        all_train_macro_lengths=dict(collections.Counter(len(demonstration_actions(r)) for r in train)),
        versions=dict(minigrid=minigrid.__version__),
        candidate_catalog='native original instruction tree plus color/type-specific native goto, pickup, open goals; excludes initially satisfied goto goals',
        support='greedy macro-action deletion with native verifier replay and physical target identity preserved; one valid sufficient subset, not unique causal support'))
    results = []
    for row in selected:
        path = OUT / 'demonstrations' / (row['item_id'] + '.json')
        if path.exists() and json.loads(path.read_text()).get('support_semantics_version') == 4:
            result = json.loads(path.read_text())
        else:
            if path.exists():
                old = json.loads(path.read_text())
                save(OUT / ('superseded_supports_v' + str(old.get('support_semantics_version', 1))) / path.name, old)
            result = add_supports(replay(row['item_id'], demonstration_actions(row), keep_trace=True))
            save(path, result)
        results.append(result)
        print(json.dumps(dict(done=len(results), item=row['item_id'], **result['support_summary'])), flush=True)
    summary = dict(status='COMPLETE', scope='TRAIN structural screen, not HSL-versus-DCHL performance',
        seconds=time.monotonic() - started, all=summarize(results),
        by_level={str(k): summarize([r for r in results if int(r['item_id'].split('_')[-1]) % 40 + 1 == k]) for k in sorted(groups)},
        oracle_prefix_caveat='Demonstration prefixes are not model failures; separate learner rollout required to estimate online supply.')
    save(OUT / 'demonstration_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
