"""Outcome-gated replay under the literal request that generated the actions.

No hindsight goal extraction is needed. Native terminal success admits the full
episode; native command admissibility masks action targets only. Earlier errors
remain in all subsequent histories. The inherited HSL FIFO samples uniformly
because every admitted episode has quality weight one.
"""
import copy
import json
from pathlib import Path


def build_result(episode):
    assert episode['status'] == 'complete'
    trajectory = episode['trajectory']
    assert trajectory and episode['steps'] == len(trajectory)
    assert episode['won'] == trajectory[-1]['won']
    steps = [{k: copy.deepcopy(s[k]) for k in
              ('step', 'observation', 'action', 'admissible_actions', 'result_observation')}
             for s in trajectory]
    assert [s['step'] for s in steps] == list(range(len(steps)))
    mask = [s['action'].lower() in {a.lower() for a in s['admissible_actions']}
            for s in steps]
    result = dict(gamefile=episode['gamefile'], requested_task_type=episode['task_type'],
                  rollout_id=episode['rollout_id'], policy_version=episode['policy_version'],
                  source_success=episode['won'], steps=steps, goals=[],
                  action_relevance_ready=True, training_ready=bool(episode['won']),
                  dialogue_hsl=True, response_contract='action_only_v1',
                  engineering_fixture_only=False, privileged_state=False,
                  relabeler='original_request_terminal_outcome_v1',
                  relevance_mode='native_admissibility_only', wall_seconds=0.,
                  relevance_worker_seconds=0.)
    if episode['won']:
        assert any(mask)
        # These fields support existing family reporting only. Admission and
        # labels depend on the original request, native won and admissibility.
        params = json.loads((Path(episode['gamefile']).parent/'traj_data.json').read_text())['pddl_params']
        destination = params['toggle_target'] if episode['task_type'] == 'look_at_obj_in_light' else params['parent_target']
        goal = dict(task_type=episode['task_type'], object_type=params['object_target'].lower(),
                    destination_type=destination.lower(), instruction=episode['task_description'],
                    original_request_bound=True, prefix_length=len(steps),
                    first_achievement_step=len(steps), action_loss_mask=mask,
                    relevant_actions=sum(mask), hsl_weight=1.,
                    relevance_rule='native_admissibility_only',
                    replay_pool='original_request_success')
        result['goals'] = [goal]
    return result
