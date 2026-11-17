"""One early fresh TRAIN attempt inside the original per-task action budget.

Both physically separate attempts retain their own canonical HSL histories.
No teacher, action override, plan, or additional optimizer update is used.
"""
import time


def collect(row, config):
    from alfworld_dialogue_hsl import collect as ordinary_collect
    began = time.monotonic()
    local_row = dict(row, guidance_selected=False)
    first_config = dict(config, correction_backend=None, _restart_stop_on_noop=True,
        record_generation_usage=True)
    first = ordinary_collect(local_row, first_config)
    if first['status'] != 'complete':
        return first
    attempts = [first]
    cut = first.get('restart_cut')
    if cut is not None:
        assert 2 <= cut <= 20 and not first['won'] and not first['trajectory'][-1]['done']
        fresh = ordinary_collect(local_row, dict(first_config,
            _restart_stop_on_noop=False, max_steps=config['max_steps']-cut,
            seed=config['seed']+cut))
        if fresh['status'] != 'complete':
            fresh['budget_restart_failed_first_attempt'] = first
            return fresh
        assert fresh['trajectory'][0]['observation'] == first['trajectory'][0]['observation']
        attempts.append(fresh)
    segments = []
    for index, attempt in enumerate(attempts):
        hindsight = attempt['hindsight']
        hindsight['attempt_index'] = index
        hindsight['source_attempt_id'] = f"{row['rollout_id']}:attempt-{index}"
        segments.append(hindsight)
    usage = [dict(u, attempt_index=index) for index, a in enumerate(attempts)
             for u in a['generation_usage']]
    total_actions = sum(a['steps'] for a in attempts)
    assert total_actions <= config['max_steps'] and len(usage) == total_actions
    result = dict(first, attempts=attempts, hindsight_segments=segments,
        won=any(a['won'] for a in attempts), reward=float(any(a['won'] for a in attempts)),
        steps=total_actions, generation_usage=usage, wall_seconds=time.monotonic()-began,
        budget_restart=dict(triggered=cut is not None, cut=cut,
            remaining_budget=config['max_steps']-cut if cut is not None else None,
            resets=len(attempts)-1, physical_attempts=len(attempts),
            neural_actions=total_actions, first_prefix_goals=len(segments[0]['goals']),
            fresh_goals=len(segments[1]['goals']) if len(segments)>1 else 0,
            teacher_calls=0, planner_calls=0, action_budget=config['max_steps'],
            histories_stitched=False))
    # Top-level hindsight is the first physical attempt, never a merged history.
    # The trainer explicitly ingests each hindsight_segments entry.
    return result
