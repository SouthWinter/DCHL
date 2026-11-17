"""Audit public witness availability in saved bounded-expert TRAIN episodes.

Literal goal-entity visibility is a heuristic information check, not a full
POMDP sufficiency certificate or an attribution of the historical decline.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/alfworld_llama1b_inline_correction75_v1'
OUT = ROOT / 'analysis/alfworld_inline_correction_v1/public_grounding'
ENTITY = re.compile(r'\b[a-z][a-z0-9_-]* \d+\b')


def visible_before_actions(trajectory):
    known, first = set(), {}
    snapshots = []
    for i, step in enumerate(trajectory):
        assert step['step'] == i
        # Read only observations available BEFORE the action, never its reply.
        for entity in ENTITY.findall(step['observation'].lower()):
            known.add(entity)
            first.setdefault(entity, i)
        snapshots.append(set(known))
    return snapshots, first


def goal_objects(goal):
    objects = {f[1] for f in goal['witness_facts']
               if f[0] == 'objecttype' and f[2] == goal['object_type'] + 'type'}
    assert len(objects) >= (2 if goal['task_type'] == 'pick_two_obj_and_place' else 1)
    return objects


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    sources, episodes, draws = {}, [], []
    episode_counts, action_counts = Counter(), Counter()
    families = defaultdict(Counter)
    for update in range(1, 76):
        folder = RUN / f'online/rounds/{update:05d}'
        rows = [json.loads(l) for l in (folder / 'rollouts.jsonl').read_text().splitlines()]
        assert len(rows) == 18 and all(r['status'] == 'complete' for r in rows)
        for row in rows:
            key = (row['gamefile'], row['policy_version'])
            assert key not in sources
            snapshots, first = visible_before_actions(row['trajectory'])
            expert = {i for i, step in enumerate(row['trajectory'])
                      if step.get('action_source') == 'environment_expert'}
            sources[key] = row, snapshots, first, expert
            solver = row.get('correction', {}).get('solver')
            if not expert:
                continue
            assert solver and solver['accepted'] and row['won']
            episode_counts['accepted_expert'] += 1
            cut = min(expert)
            planned_objects = set()
            for action in solver['plan']:
                match = re.fullmatch(r'(?:take|move|heat|cool|clean) (.+) (?:from|to|with) .+', action)
                if match:
                    planned_objects.add(match[1])
            unseen = sorted(planned_objects - snapshots[cut])
            episode_counts['plan_has_object_not_observed_at_cut'] += bool(unseen)
            family = row['task_type']
            families[family]['accepted_expert'] += 1
            families[family]['plan_has_object_not_observed_at_cut'] += bool(unseen)
            episodes.append(dict(update=update, task=row['gamefile'], family=family,
                cut=cut, planned_objects=sorted(planned_objects),
                objects_not_observed_at_cut=unseen,
                first_observed_before_action={o: first.get(o) for o in sorted(planned_objects)},
                teacher_actions=len(expert), raw_request=row['task_description']))
        provenance = json.loads((folder / 'batch_provenance.json').read_text())
        for p in provenance:
            if p['source'] != 'hindsight':
                continue
            row, snapshots, first, expert = sources[(p['task'], p['policy_version'])]
            goals = [g for g in row['hindsight']['goals']
                     if g['instruction'] == p['goal'] and g['prefix_length'] == p['prefix_length']]
            assert len(goals) == 1
            goal = goals[0]
            active = {i for i, yes in enumerate(goal['action_loss_mask']) if yes}
            assert len(active) == p['supervised_actions']
            witnesses = goal_objects(goal)
            teacher = sorted(active & expert)
            unavailable = [i for i in teacher if not witnesses <= snapshots[i]]
            student = sorted(active - expert)
            weight = .3 / 32 / goal['prefix_length']
            action_counts['selected_actions'] += len(active)
            action_counts['selected_teacher_actions'] += len(teacher)
            action_counts['selected_teacher_without_all_goal_objects_visible'] += len(unavailable)
            action_counts['selected_student_actions'] += len(student)
            verbs = Counter(row['trajectory'][i]['action'].split()[0] for i in unavailable)
            for verb, n in verbs.items():
                action_counts['unavailable_teacher_' + verb] += n
            families[goal['task_type']]['selected_teacher_actions'] += len(teacher)
            families[goal['task_type']]['selected_teacher_without_all_goal_objects_visible'] += len(unavailable)
            if teacher:
                draws.append(dict(update=update, draw=p['sampling_draw_id'],
                    family=goal['task_type'], task=p['task'], source_version=p['policy_version'],
                    instruction=goal['instruction'], prefix_length=goal['prefix_length'],
                    goal_witness_objects=sorted(witnesses), teacher_actions=teacher,
                    teacher_without_all_goal_objects_visible=unavailable,
                    unavailable_verbs=dict(verbs), per_action_coefficient=weight))
    assert len(sources) == 1350
    report = dict(status='PASS', updates=75, source_episodes=1350,
        episodes=dict(episode_counts), selected_actions=dict(action_counts),
        by_family={k: dict(v) for k, v in families.items()},
        scope='Actual saved TRAIN plans, public observations before each action, and exact submitted HSL draws; literal entity IDs, not full observability, causality or learning value.',
        candidate_operation='Keep all original histories/student action labels; examine masking expert targets until goal witness objects are publicly available and binding verified original-goal inputs to the actual request. This is an untrained hypothesis.',
        new_updates=0, new_model_forwards=0, new_environment_executions=0,
        cpu_wall_seconds=time.monotonic()-started)
    for name, data in [('results.json', report), ('episodes.json', episodes), ('draws.json', draws)]:
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'by_family'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
