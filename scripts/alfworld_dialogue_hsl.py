"""Full-dialogue collection and action-only hindsight targets for the ETO start."""
import copy
import json
import os
import re
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def initial_messages(observation):
    if os.environ.get('ALFWORLD_ACTION_CONTRACT')=='1':
        from alfworld_action_contract import prefix
        return prefix()+[dict(role='user',content=observation)]
    prefix=json.loads((ROOT/'data/alfworld_eto_sft800_v1/instruction.json').read_text())
    return [dict(role={'human':'user','gpt':'assistant'}[m['from']],content=m['value']) for m in prefix]+[dict(role='user',content=observation)]

def api(endpoint,route,payload):
    req=urllib.request.Request(endpoint+route,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as response:return json.load(response)

def context_messages(endpoint,messages):
    kept=list(messages)
    while True:
        encoded=api(endpoint,'/tokenize',dict(model='qwen3-4b',messages=kept,add_generation_prompt=True))
        if encoded['count']<=7680:
            if os.environ.get('ALFWORLD_RECORD_POLICY_CONTEXT') == '1':
                global LAST_POLICY_INPUT_IDS
                LAST_POLICY_INPUT_IDS=encoded['tokens']
            return kept
        if len(kept)<=5:raise ValueError('Initial/latest context too long')
        del kept[3:5]

def raw_action(response):
    match=re.search(r'(?:^|\n)Action:\s*([^\n]+)',response)
    return match.group(1).strip() if match else ''

def environment_action(action,admissible):
    match=re.fullmatch(r'put (.+) in/on (.+)',action)
    mapped=f'move {match[1]} to {match[2]}' if match else action
    return mapped if mapped in admissible else action

def public_action(action):
    match=re.fullmatch(r'move (.+) to (.+)',action)
    return f'put {match[1]} in/on {match[2]}' if match else action

def hindsight_records(rollout,goal,mixture,count):
    from alfworld_hindsight import replace_instruction
    action_contract=rollout.get('response_contract')=='action_only_v1'
    steps=rollout['steps'][:goal['prefix_length']]
    history=initial_messages(replace_instruction(steps[0]['observation'],goal['instruction']))
    if action_contract:
        from alfworld_action_contract import prefix
        history=prefix()+history[-1:]
    records=[]
    for i,step in enumerate(steps):
        preserve=not action_contract and rollout.get('dialogue_supervision')=='preserve_response_action_span'
        if preserve:
            response=step['response']
            match=re.search(r'(?:^|\n)(Action:\s*[^\n]+)',response)
            if goal['action_loss_mask'][i] and (not match or environment_action(raw_action(response),step['admissible_actions'])!=step['action']):
                raise ValueError('Supervised raw response does not match executed action')
            target=dict(role='assistant',content=response)
            if match:target['loss_char_start']=match.start(1)
        else:
            target=dict(role='assistant',content='Action: '+public_action(step['action']))
        msgs=copy.deepcopy(history+[target])
        for m in msgs:m['loss_on_this_turn']=False
        msgs[-1]['loss_on_this_turn']=bool(goal['action_loss_mask'][i])
        records.append(dict(messages=msgs,step=i,action=step['action'],enable_thinking=False,
            policy_version=rollout['policy_version'],source_episode_id=rollout['gamefile'],goal_task_type=goal['task_type'],action_supervised=bool(goal['action_loss_mask'][i]),
            trajectory_action_coefficient=mixture/count/len(steps)))
        if 'source_attempt_id' in rollout:
            records[-1]['source_attempt_id'] = rollout['source_attempt_id']
        history.extend([target,dict(role='user',content='Observation: '+replace_instruction(step['result_observation'],goal['instruction']))])
    return records

def collect(row,config):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
    from alfworld_hindsight import replay_and_relabel
    start=time.monotonic();result=dict(row,trajectory=[],status='running',dialogue_hsl=True,policy_version=config['policy_version'])
    env=None
    guidance = None
    correction = None
    correction_plan = []
    acquisition_mode = config.get('correction_backend') == 'public_acquisition_planner'
    operation_mode = config.get('correction_backend') == 'goal_operation_label'
    label_mode = config.get('correction_backend') in ('cost_screened_label','goal_operation_label')
    operation_context = None
    label_probe = None
    correction_labels = []
    public_entities = set()
    acquisition_condition = None
    usage_rows = []
    try:
        with urllib.request.urlopen(config['endpoint']+'/hsl/policy_version',timeout=30) as response:version=json.load(response)
        if version['policy_version']!=config['policy_version'] or version['pending_version'] is not None:raise ValueError('Policy version mismatch')
        restart_stop = bool(config.get('_restart_stop_on_noop', False))
        noop_errors = 0
        env=textworld.start(row['gamefile'],request_infos=textworld.EnvInfos(won=True,admissible_commands=True,facts=restart_stop),wrappers=[AlfredDemangler(shuffle=False)])
        state=env.reset();messages=initial_messages(str(state.feedback))
        result['task_description']=str(state.feedback).split('Your task is to: ')[-1].split('\n')[0]
        if acquisition_mode or label_mode:
            from alfworld_inline_correction import request_condition
            acquisition_condition = request_condition(row)
        if operation_mode and row.get('guidance_selected',False) and acquisition_condition['status']=='supported':
            from alfworld_goal_opportunity import grounded_goal_context
            operation_context = grounded_goal_context(env,sas_grounded=True)
        condition_mode = config.get('correction_backend') == 'equivalent_condition'
        semantic_condition = None
        if condition_mode:
            from alfworld_inline_correction import request_condition
            from alfworld_hindsight import replace_instruction
            semantic_condition = request_condition(row)
            changed = row.get('guidance_selected',False) and semantic_condition['status']=='supported'
            if changed:
                # Change only this episode's initial task, not few-shot examples
                # or any environmental observation. All actions remain student
                # generated, with the ordinary per-step seed and40-step budget.
                messages[-1]['content'] = replace_instruction(messages[-1]['content'],semantic_condition['instruction'])
            result['condition_transfer'] = dict(selected=row.get('guidance_selected',False),
                activated=bool(changed), original_instruction=result['task_description'],
                actor_instruction=semantic_condition['instruction'] if changed else result['task_description'],
                semantic_condition=semantic_condition, new_teacher_model_calls=0, extra_environment_steps=0)
        for i in range(config['max_steps']):
            observation=str(state.feedback);admissible=list(state.admissible_commands)
            active = not condition_mode and row.get('guidance_selected', False) and (operation_mode or i >= row['guidance_cut'])
            planner_mode = config.get('correction_backend') in ('bounded_planner', 'public_grounded_planner', 'public_acquisition_planner')
            planner_ready = True
            if acquisition_mode or label_mode:
                from alfworld_acquisition_correction import observe_entities, public_goal_candidates, plan_public_support
                observe_entities(observation, public_entities)
                planner_ready = public_goal_candidates(acquisition_condition, public_entities)
            if active and planner_mode and correction is None and planner_ready:
                from alfworld_inline_correction import bounded_plan
                correction = bounded_plan(env, config.get('planner_timeout', 5))
                if acquisition_mode:
                    correction.update(public_support=plan_public_support(correction['plan'], public_entities),
                                      public_handoff_step=i)
                public_supported = not acquisition_mode or correction['public_support']['supported']
                if correction['status'] == 'complete' and public_supported and 0 < len(correction['plan']) <= config['max_steps']-i:
                    correction_plan = list(correction['plan'])
                    correction['accepted'] = True
                else:
                    correction['accepted'] = False
                correction['teacher_actions'] = 0
            if active and label_mode and not operation_mode and label_probe is None and planner_ready:
                from alfworld_cost_screened_labels import before_action
                label_probe = before_action(env, i, public_entities, admissible,
                    config['max_steps'], config.get('planner_timeout', 5))
            check_label_now = label_mode and label_probe is not None and label_probe['step'] == i
            if active and not planner_mode and not label_mode and guidance is None:
                from alfworld_inline_correction import guided_messages
                _, guidance = guided_messages(messages,result['task_description'],observation,
                                               config['guidance_reference_bank'])
                reference = next(e for e in config['guidance_reference_bank']
                                 if e['id'] == guidance['reference_id'])
            current = messages
            if active and not planner_mode and not label_mode:
                current = copy.deepcopy(messages)
                current[0]['content'] += reference['hint']
            teacher_action = None
            if correction_plan:
                candidate = correction_plan.pop(0)
                if candidate in admissible:
                    teacher_action = candidate
                else:
                    correction_plan = []
                    correction['execution_fallback'] = 'plan_action_not_admissible'
            if teacher_action is not None:
                action=teacher_action;raw=public_action(action);response='Action: '+raw
                correction['teacher_actions'] += 1
            else:
                prompt=context_messages(config['endpoint'],current)
                payload=dict(model='qwen3-4b',messages=prompt,temperature=float(config.get('temperature',1.)),top_p=1.,max_tokens=512,seed=config['seed']+row['rollout_id']*1009+i,chat_template_kwargs={'enable_thinking':False})
                if os.environ.get('ALFWORLD_RECORD_POLICY_CONTEXT') == '1':
                    payload.update(logprobs=True,return_tokens_as_token_ids=True)
                out=api(config['endpoint'],'/v1/chat/completions',payload)
                if config.get('record_generation_usage', False):
                    usage = out['usage']
                    usage_rows.append(dict(step=i,guided=bool(active and not planner_mode and not label_mode),
                        prompt_tokens=int(usage['prompt_tokens']),
                        completion_tokens=int(usage['completion_tokens'])))
                response=out['choices'][0]['message']['content'] or ''
                raw=raw_action(response);action=environment_action(raw,admissible) or 'invalid action'
            if active and operation_mode and label_probe is None and planner_ready:
                from alfworld_goal_operation_correction import before_student_action
                label_probe = before_student_action(env,state,acquisition_condition,public_entities,
                    operation_context,action,prompt,i,config['max_steps'],config.get('planner_timeout',5))
                check_label_now = label_probe is not None
            if restart_stop:
                from alfworld_hindsight import facts_of
                before_facts = facts_of(state)
            state,_,done=env.step(action)
            if restart_stop:
                no_change = facts_of(state) == before_facts
                invalid = action.lower() not in {x.lower() for x in admissible}
                noop_errors = noop_errors + 1 if invalid and no_change else 0
            if check_label_now:
                observe_entities(str(state.feedback), public_entities)
                if operation_mode:
                    from alfworld_goal_operation_correction import after_student_action
                    label = after_student_action(env,label_probe,bool(state.won),config.get('planner_timeout',5),
                        prompt,result,result['task_description'])
                else:
                    from alfworld_cost_screened_labels import after_action
                    label = after_action(env, label_probe, action, prompt, public_entities,
                        bool(state.won), config.get('planner_timeout', 5), result, result['task_description'])
                if label is not None:
                    correction_labels.append(label)
            result['trajectory'].append(dict(step=i,observation=observation,admissible_actions=admissible,action=action,raw_action=raw,response=response,result_observation=str(state.feedback),won=bool(state.won),done=bool(done)))
            if planner_mode:
                result['trajectory'][-1]['action_source'] = 'environment_expert' if teacher_action is not None else 'student'
            if teacher_action is None and os.environ.get('ALFWORLD_RECORD_POLICY_CONTEXT') == '1':
                from datetime import datetime, timezone
                result['trajectory'][-1].update(policy_messages=copy.deepcopy(prompt),
                    policy_template_date=os.environ.get('ALFWORLD_FIXED_TEMPLATE_DATE') or datetime.now(timezone.utc).strftime('%d %b %Y'),
                    policy_input_ids=list(LAST_POLICY_INPUT_IDS),
                    policy_response_ids=[int(t['token'].removeprefix('token_id:'))
                        for t in out['choices'][0]['logprobs']['content']])
            history_response=('Action: '+public_action(action)) if os.environ.get('ALFWORLD_ACTION_CONTRACT')=='1' else response
            messages.extend([dict(role='assistant',content=history_response),dict(role='user',content='Observation: '+str(state.feedback))])
            if done or state.won:break
            if restart_stop and noop_errors >= 2 and i + 1 <= 20:
                result['restart_cut'] = i + 1
                result['restart_trigger'] = dict(consecutive_invalid=2, native_fact_delta=0,
                    command_identity_required=False)
                break
        result.update(status='complete',won=bool(state.won),reward=float(state.won),steps=len(result['trajectory']))
        if config.get('record_generation_usage', False):
            result['generation_usage'] = usage_rows
        if config.get('guidance_reference_bank'):
            result['guidance'] = dict(selected=row.get('guidance_selected',False),
                cut=row.get('guidance_cut'),activated=guidance is not None,
                guided_actions=sum(r['guided'] for r in usage_rows),
                reference=guidance,training_history_has_reference=False)
        if planner_mode:
            result['correction'] = dict(selected=row.get('guidance_selected',False),
                cut=row.get('guidance_cut'), activated=correction is not None, solver=correction,
                original_goal_verified=bool(state.won), extra_environment_steps=0,
                new_teacher_model_calls=0)
        if config.get('skip_hindsight_relabel',False):
            return result
        observable = os.environ.get('ALFWORLD_HINDSIGHT_BACKEND') == 'observable_evidence_v1'
        original_replay = os.environ.get('ALFWORLD_ORIGINAL_REQUEST_REPLAY') == '1'
        if original_replay:
            from alfworld_original_request_replay import build_result
            result['hindsight'] = build_result(result)
        elif observable:
            from alfworld_execution_evidence import relabel
            result['hindsight']=relabel(result)
        else:
            result['hindsight']=replay_and_relabel(result,config.get('relevance_rule','backward'))
        result['hindsight'].update(dialogue_hsl=True,engineering_fixture_only=False)
        if os.environ.get('ALFWORLD_ACTION_CONTRACT')=='1':
            from alfworld_action_contract import mask_repeated_navigation
            h=result['hindsight']
            if not observable and not original_replay:
                h['goals']=[mask_repeated_navigation(h['steps'],g) for g in h['goals']]
            h['response_contract']='action_only_v1'
        if os.environ.get('ALFWORLD_DIALOGUE_SUPERVISION')=='preserve_response_action_span':
            result['hindsight']['dialogue_supervision']='preserve_response_action_span'
            for target,source in zip(result['hindsight']['steps'],result['trajectory'],strict=True):
                target['response']=source['response']
        if condition_mode:
            from alfworld_inline_correction import bind_original_request
            result['condition_transfer']['bound_goals'] = bind_original_request(result, semantic_condition)
        if config.get('correction_backend') == 'public_grounded_planner':
            from alfworld_public_correction import project_correction
            project_correction(result)
        if acquisition_mode:
            from alfworld_inline_correction import bind_original_request
            bound = bind_original_request(result, acquisition_condition)
            result['acquisition_correction'] = dict(bound_goals=bound,
                planner_called=int(correction is not None),
                public_plan_accepted=int(bool(correction and correction['accepted'])),
                handoff_step=correction.get('public_handoff_step') if correction else None,
                student_prefix_masks_preserved=True, extra_environment_steps=0,
                teacher_llm_calls=0, extra_optimizer_updates=0)
        if label_mode:
            result['action_labeling'] = dict(selected=row.get('guidance_selected', False),
                cut=0 if operation_mode else row.get('guidance_cut'), probe=label_probe,
                accepted_labels=len(correction_labels), executed_expert_actions=(
                    label_probe['teacher_solver'].get('expert_executed_actions') if operation_mode and label_probe else 0),
                extra_environment_steps=0, teacher_llm_calls=0)
            if operation_mode:
                result['action_labeling']['candidate_branch_attempts'] = int(label_probe is not None)
                result['action_labeling']['mode'] = 'goal_operation_label'
                result['hindsight']['goal_operation_labels'] = correction_labels
            else:
                result['hindsight']['cost_screened_labels'] = correction_labels
    except Exception as exc:result.update(status='infrastructure_error',error=f'{type(exc).__name__}: {exc}')
    finally:
        if config.get('record_generation_usage', False):
            result['generation_usage'] = usage_rows
        if env is not None:env.close()
    result['wall_seconds']=time.monotonic()-start
    return result
