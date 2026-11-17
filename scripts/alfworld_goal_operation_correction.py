"""Original-goal operation alternatives verified against the actual student.

No teacher LLM or neural alternative rollout. One native action fork per
preselected episode; up to two bounded native planning queries. A feasible
remaining-plan advantage is a supervision proxy, not an optimal Q estimate.
"""
import copy
from alfworld_goal_opportunity import available_grounded_goal_operations
from alfworld_inline_correction import bounded_plan
from alfworld_acquisition_correction import observe_entities,plan_public_support


def before_student_action(env,state,condition,known,context,actual_action,prompt,step,max_steps,timeout):
    operations=available_grounded_goal_operations(env,state,condition,known,context)
    candidates=set(operations)-{actual_action}
    if not candidates:
        return None
    target=sorted(candidates)[0]
    teacher=bounded_plan(env,timeout,action=target)
    support=plan_public_support([target]+teacher['plan'],known)
    input_entities=set()
    for message in prompt: observe_entities(message['content'],input_entities)
    input_support=plan_public_support([target]+teacher['plan'],input_entities)
    accepted=(teacher['status']=='complete' and (teacher.get('won') or bool(teacher['plan']))
        and support['supported'] and input_support['supported'] and 1+len(teacher['plan'])<=max_steps-step)
    return dict(step=step,expert_action=target,actual_student_action=actual_action,
        available_operations=operations,teacher_solver=teacher,teacher_support=support,
        actual_input_support=input_support,teacher_accepted=bool(accepted),accepted=False,
        candidate_branch_attempts=1,reason='target_branch_not_supported' if not accepted else 'await_actual_student_action')


def after_student_action(env,probe,won,timeout,prompt,row,task):
    if not probe['teacher_accepted']:
        return None
    student=dict(status='complete',plan=[],seconds=0.,won=True) if won else bounded_plan(env,timeout)
    complete=student['status']=='complete' and (won or bool(student['plan']))
    advantage=len(student['plan'])-len(probe['teacher_solver']['plan']) if complete else None
    accepted=bool(complete and advantage>0)
    probe.update(student_after_solver=student,student_query_called=not won,
        feasible_plan_advantage=advantage,accepted=accepted,
        reason='positive_goal_operation_advantage' if accepted else 'no_positive_complete_branch_advantage')
    if not accepted:
        return None
    from alfworld_dialogue_hsl import public_action
    messages=copy.deepcopy(prompt)
    for message in messages: message['loss_on_this_turn']=False
    target=probe['expert_action']
    messages.append(dict(role='assistant',content='Action: '+public_action(target),loss_on_this_turn=True))
    return dict(messages=messages,step=probe['step'],action=target,enable_thinking=False,
        policy_version=row['policy_version'],source_episode_id=row['gamefile'],
        goal_task_type=row['task_type'],action_supervised=True,original_request=task,
        prefix_length=probe['step']+1,feasible_plan_advantage=advantage,
        student_action=probe['actual_student_action'],label_kind='native_original_goal_operation',
        operation_effects=probe['available_operations'][target],candidate_branch_attempts=1,
        executed_expert_actions=probe['teacher_solver'].get('expert_executed_actions'))
