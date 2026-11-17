"""Opt-in independent task and row-order streams for matched online experiments."""
import random
import json
import os
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=4)
def _frozen_document(path):
    document = json.loads(Path(path).read_text())
    task_key = document.get('task_key', 'gamefile')
    if task_key not in ('gamefile', 'id'):
        raise ValueError('Unsupported frozen task identity key')
    rounds = document['rounds']
    assert [r['step'] for r in rounds] == list(range(1, document['updates'] + 1))
    assert all(len(r['tasks']) == document['rollouts_per_update'] for r in rounds)
    assert all(len({t[task_key] for t in r['tasks']}) == len(r['tasks']) for r in rounds)
    return document


def frozen_schedule_state():
    path = os.environ.get('ALFWORLD_FROZEN_TASK_SCHEDULE')
    if not path:
        return None
    # Preserve the complete schedule for exact resume comparison without hashes.
    return dict(path=str(Path(path).resolve()), document=_frozen_document(path))


def round_tasks(tasks,count,seed,step,legacy_rng,stable=False):
    frozen = frozen_schedule_state()
    if frozen is not None:
        document = frozen['document']
        if not stable or seed != document['seed'] or count != document['rollouts_per_update']:
            raise ValueError('Frozen task schedule seed/count/stable-mode mismatch')
        if not 1 <= step <= document['updates']:
            raise ValueError('Update outside frozen task schedule')
        selected = document['rounds'][step - 1]['tasks']
        task_key = document.get('task_key', 'gamefile')
        pool = {r[task_key]: r for r in tasks}
        if not all(pool.get(r[task_key]) == r for r in selected):
            raise ValueError('Frozen task is not present in the configured TRAIN pool')
        return [dict(r) for r in selected]
    rng=random.Random(seed+step*1000003) if stable else legacy_rng
    return rng.sample(tasks,count)


def row_rng(seed,step,legacy_rng,stable=False):
    return random.Random(seed+step*1000033) if stable else legacy_rng


def completed_rows(rows,count,stable=False):
    completed={r['rollout_id']:r for r in rows if r['status']=='complete'}
    if len(completed)!=count:raise ValueError('online round is missing completed rollouts')
    if stable:
        if set(completed)!=set(range(count)):raise ValueError('Unexpected rollout identities')
        return [completed[i] for i in range(count)]
    return list(completed.values())


def check_resume_schedule(state,stable):
    if state.get('stable_data_schedule',False)!=stable:
        raise ValueError('Cannot change data schedule semantics on optimizer resume')
    if state.get('frozen_task_schedule') != frozen_schedule_state():
        raise ValueError('Cannot change the frozen task schedule on optimizer resume')
