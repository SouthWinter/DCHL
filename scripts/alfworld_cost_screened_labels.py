"""Sparse original-goal labels screened by native feasible-plan overhead.

The planner uses privileged state/rules. Plan length is a proxy, not optimal Q.
Actual environment trajectories and ordinary HSL masks remain student generated.
"""
from collections import deque
import copy
import random

from alfworld_hsl_buffer import HSLBuffer
from alfworld_inline_correction import bounded_plan
from alfworld_acquisition_correction import observe_entities, plan_public_support


def before_action(env, step, known, admissible, max_steps, timeout):
    solver = bounded_plan(env, timeout)
    support = plan_public_support(solver['plan'], known)
    supported = (solver['status'] == 'complete' and support['supported']
                 and 0 < len(solver['plan']) <= max_steps - step
                 and solver['plan'][0] in admissible)
    return dict(step=step, before_solver=solver, before_support=support,
                before_accepted=supported, before_admissible=list(admissible))


def after_action(env, probe, actual_action, prompt, known, won, timeout, row, task):
    probe.update(actual_student_action=actual_action, accepted=False)
    if not probe['before_accepted']:
        probe['reason'] = 'before_plan_not_supported'
        return None
    if won:
        after = dict(status='complete', plan=[], seconds=0.)
    else:
        after = bounded_plan(env, timeout)
    after_support = plan_public_support(after['plan'], known)
    prompt_entities = set()
    for message in prompt:
        observe_entities(message['content'], prompt_entities)
    input_support = plan_public_support(probe['before_solver']['plan'], prompt_entities)
    complete = after['status'] == 'complete' and (won or bool(after['plan']))
    gap = 1 + len(after['plan']) - len(probe['before_solver']['plan']) if complete else None
    target = probe['before_solver']['plan'][0]
    probe.update(after_solver=after, after_query_called=not won, after_support=after_support,
                 actual_input_support=input_support, feasible_plan_overhead=gap,
                 expert_action=target)
    if not complete or not after_support['supported'] or not input_support['supported']:
        probe['reason'] = 'after_plan_or_actual_input_not_supported'
        return None
    if gap <= 0 or target == actual_action:
        probe['reason'] = 'no_positive_supported_disagreement'
        return None
    from alfworld_dialogue_hsl import public_action
    messages = copy.deepcopy(prompt)
    for message in messages:
        message['loss_on_this_turn'] = False
    messages.append(dict(role='assistant', content='Action: ' + public_action(target), loss_on_this_turn=True))
    probe.update(accepted=True, reason='positive_feasible_plan_overhead')
    return dict(messages=messages, step=probe['step'], action=target, enable_thinking=False,
        policy_version=row['policy_version'], source_episode_id=row['gamefile'],
        goal_task_type=row['task_type'], action_supervised=True,
        original_request=task, prefix_length=probe['step'] + 1,
        feasible_plan_overhead=gap, student_action=actual_action,
        label_kind='native_original_goal_action', executed_expert_actions=0)


class CostScreenedBuffer(HSLBuffer):
    def __init__(self, capacity=100, seed=42, allow_engineering=False):
        super().__init__(capacity, seed, allow_engineering)
        self.labels = deque(maxlen=capacity)
        self.label_rng = random.Random(seed + 991739)

    def append_round(self, results, policy_version):
        super().append_round(results, policy_version)
        for result in results:
            for label in result.get('goal_operation_labels', result.get('cost_screened_labels', [])):
                margin = label.get('feasible_plan_advantage',label.get('feasible_plan_overhead'))
                assert label['policy_version'] == policy_version and margin > 0
                assert sum(bool(m.get('loss_on_this_turn')) for m in label['messages']) == 1
                entry = copy.deepcopy(label)
                entry['collection_round'] = self.round_index - 1
                self.labels.append(entry)
        # Match the existing ordinary-source freshness horizon, rather than
        # replaying a small early-label pool throughout the entire75 updates.
        oldest = min((e['collection_round'] for e in self.demonstrations), default=self.round_index - 1)
        self.labels = deque((e for e in self.labels if e['collection_round'] >= oldest), maxlen=self.capacity)

    def state_dict(self):
        state = super().state_dict()
        state.update(cost_screened_labels=list(self.labels), label_rng=self.label_rng.getstate())
        return state

    @classmethod
    def from_state_dict(cls, state):
        buffer = super().from_state_dict(state)
        buffer.labels.extend(state['cost_screened_labels'])
        def tuples(value):
            return tuple(map(tuples, value)) if isinstance(value, (list, tuple)) else value
        buffer.label_rng.setstate(tuples(state['label_rng']))
        return buffer


def mixed_records(demonstrations, buffer, rng, trajectories=32, hindsight_fraction=.3,
                  original_tasks=None, goal_logits=None):
    from alfworld_hsl_batch import mixed_records as ordinary
    rows, provenance = ordinary(demonstrations, buffer, rng, trajectories,
        hindsight_fraction, original_tasks, goal_logits)
    assert trajectories == 32 and goal_logits is None
    ordinary_draws = [p for p in provenance if p['source'] == 'hindsight']
    count = min(4, len(buffer.labels), len(ordinary_draws))
    if not count:
        return rows, provenance
    replaced = {p['sampling_draw_id'] for p in ordinary_draws[-count:]}
    rows = [r for r in rows if r.get('sampling_draw_id', -1) not in replaced]
    provenance = [p for p in provenance if p.get('sampling_draw_id', -1) not in replaced]
    for draw, label in zip(sorted(replaced), buffer.label_rng.sample(list(buffer.labels), count)):
        record = copy.deepcopy(label)
        record.update(sampling_draw_id=draw,
            trajectory_action_coefficient=hindsight_fraction / trajectories / label['prefix_length'])
        rows.append(record)
        provenance.append(dict(source='original_goal_correction', task=label['source_episode_id'],
            sampling_draw_id=draw, goal_task_type=label['goal_task_type'], goal=label['original_request'],
            prefix_length=label['prefix_length'], supervised_actions=1,
            policy_version=label['policy_version'], step=label['step'],
            expert_action=label['action'], student_action=label['student_action'],
            collection_round=label['collection_round'],
            trajectory_action_coefficient=record['trajectory_action_coefficient'],
            **({'feasible_plan_advantage':label['feasible_plan_advantage']} if 'feasible_plan_advantage' in label
               else {'feasible_plan_overhead':label['feasible_plan_overhead']})))
    assert len(provenance) == 64
    return rows, provenance
