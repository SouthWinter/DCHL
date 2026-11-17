"""Project bounded expert corrections onto the student's public task history.

Keep the original executed histories and sampling weights. Original-goal
corrections teach the verified expert suffix under the actual request; expert
targets lacking public witness identities are masked. The all-goal-witness
gate is a heuristic; a current action can require only part of the goal's
entities. It certifies neither full POMDP sufficiency nor independent learning.
"""
from collections import Counter
import copy

from alfworld_inline_correction import request_condition, bind_original_request
from audit_alfworld_expert_observability import visible_before_actions, goal_objects


def project_correction(result):
    history = result['hindsight']
    before, _ = visible_before_actions(result['trajectory'])
    expert = {i for i, step in enumerate(result['trajectory'])
              if step.get('action_source') == 'environment_expert'}
    if expert:
        solver = result['correction']['solver']
        assert solver['accepted']
    condition = request_condition(result)
    bound = bind_original_request(result, condition)
    totals = Counter(bound_goals=bound, expert_actions=len(expert))
    trainable, excluded = [], []
    for goal in history['goals']:
        original_goal = copy.deepcopy(goal)
        old = list(goal['action_loss_mask'])
        witnesses = goal_objects(goal)
        original_correction = bool(expert and goal.get('original_request_bound'))
        mask, reasons = list(old), list(goal['relevance_reasons'])
        counters = Counter()
        for i, yes in enumerate(old):
            if not yes:
                continue
            if original_correction and i not in expert:
                mask[i] = False
                reasons[i] = 'student_prefix_input_for_original_goal_correction'
                counters['original_correction_prefix_masked'] += 1
            elif i in expert and not witnesses <= before[i]:
                mask[i] = False
                reasons[i] = 'expert_target_before_public_goal_witnesses'
                counters['expert_targets_without_public_witnesses_masked'] += 1
        count = sum(mask)
        # Keep HSL's prior sampling distribution and prefix-length coefficient;
        # masking removes targets rather than reallocating their scalar mass.
        goal.update(action_loss_mask=mask, relevance_reasons=reasons,
                    relevant_actions=count,
                    public_correction=dict(original_correction=original_correction,
                        original_relevant_actions=sum(old), kept_actions=count,
                        witness_objects=sorted(witnesses),
                        original_sampling_weight=goal['hsl_weight'], **dict(counters)))
        totals.update(counters)
        totals['original_correction_goals'] += original_correction
        totals['kept_action_targets'] += count
        if count:
            trainable.append(goal)
        else:
            # Preserve unique verified evidence, but a zero-target entry has no
            # training role and must not occupy a sampled demonstration slot.
            excluded.append(dict(original_goal=original_goal, projected_goal=goal))
            totals['zero_target_goals_excluded'] += 1
    history['goals'] = trainable
    history['public_correction_excluded_goals'] = excluded
    result['public_correction'] = dict(totals, teacher_llm_calls=0,
        extra_environment_steps=0, extra_optimizer_updates=0,
        witness_scope='Literal object identities present in preceding public observations; not full state sufficiency')
    history['public_correction'] = result['public_correction']
    return result
