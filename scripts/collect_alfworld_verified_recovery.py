"""Replay learner prefix exactly, then generate reference-assisted recovery."""
import argparse,copy,json,os,time,urllib.request
from multiprocessing import get_context
from pathlib import Path
from alfworld_dialogue_hsl import initial_messages,public_action,environment_action,raw_action,context_messages,api
CONFIG={}
def init(config):
    CONFIG.update(config);os.environ['ALFWORLD_DATA']=config['data_root'];os.environ['ALFWORLD_ACTION_CONTRACT']='1'

def execute(job,forced_actions=None):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
    row=job['row'];cut=job['cut'];start=time.monotonic();env=None
    out={k:row[k] for k in ('gamefile','rollout_id','policy_version','task_type','task_description')}
    out.update(cut=cut,round=job['round'],attempt=job['attempt'],trajectory=[],won=False,status='running',generated_steps=0,replayed_steps=0,reference_id=job['reference']['id'])
    usage_rows=[]
    try:
        if forced_actions is None:
            with urllib.request.urlopen(CONFIG['endpoint']+'/hsl/policy_version',timeout=30) as response:version=json.load(response)
            if version['policy_version']!=row['policy_version'] or version['pending_version'] is not None:raise ValueError('Recovery policy version mismatch')
        env=textworld.start(row['gamefile'],request_infos=textworld.EnvInfos(won=True,admissible_commands=True),wrappers=[AlfredDemangler(shuffle=False)])
        state=env.reset();messages=initial_messages(str(state.feedback))
        for source in row['trajectory'][:cut]:
            if str(state.feedback)!=source['observation']:raise ValueError('Prefix observation mismatch')
            state,_,done=env.step(source['action']);out['replayed_steps']+=1
            if str(state.feedback)!=source['result_observation']:raise ValueError('Prefix transition mismatch')
            if done or state.won:raise ValueError('Prefix already terminal')
            out['trajectory'].append(copy.deepcopy(source))
            messages.extend([dict(role='assistant',content='Action: '+public_action(source['action'])),dict(role='user',content='Observation: '+str(state.feedback))])
        # Reference is confined to data generation, never copied into training history.
        hint='\nA successful example from another episode follows. Adapt it to the current observations and original task; object identifiers and locations may differ.\n'+job['reference']['text']
        guided=copy.deepcopy(messages);guided[0]['content']+=hint
        for j in range(int(job.get('max_steps',12))):
            observation=str(state.feedback);admissible=list(state.admissible_commands)
            if forced_actions is not None:
                if j>=len(forced_actions):break
                action=forced_actions[j];response='Action: '+public_action(action)
            else:
                # Optional handoff: the ordinary history never contains reference text.
                prompt=context_messages(CONFIG['endpoint'],guided if j<int(job.get('reference_steps',job.get('max_steps',12))) else messages)
                result=api(CONFIG['endpoint'],'/v1/chat/completions',dict(model='qwen3-4b',messages=prompt,temperature=1.,top_p=1.,max_tokens=job.get('max_tokens',128),
                    seed=CONFIG['seed']+job['round']*100003+row['rollout_id']*1009+job['attempt']*503+j+job.get('seed_offset',0),chat_template_kwargs={'enable_thinking':False}))
                if job.get('record_generation_usage'):
                    usage_rows.append(dict(step=cut+j,guided=j<int(job.get('reference_steps',job.get('max_steps',12))),
                        prompt_tokens=result['usage']['prompt_tokens'],completion_tokens=result['usage']['completion_tokens']))
                response=result['choices'][0]['message']['content'] or ''
                action=environment_action(raw_action(response),admissible) or 'invalid action'
            state,_,done=env.step(action);out['generated_steps']+=1
            out['trajectory'].append(dict(step=cut+j,observation=observation,action=action,result_observation=str(state.feedback),won=bool(state.won),done=bool(done)))
            guided.extend([dict(role='assistant',content='Action: '+public_action(action)),dict(role='user',content='Observation: '+str(state.feedback))])
            messages.extend([dict(role='assistant',content='Action: '+public_action(action)),dict(role='user',content='Observation: '+str(state.feedback))])
            if state.won or done:break
        out.update(status='complete',won=bool(state.won),original_goal_verified=bool(state.won))
    except Exception as exc:out.update(status='infrastructure_error',error=f'{type(exc).__name__}: {exc}')
    finally:
        if job.get('record_generation_usage'):
            out.update(generation_usage=usage_rows,condition=job['condition'])
        if env is not None:env.close()
    out['seconds']=time.monotonic()-start
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',required=True);p.add_argument('--output',required=True);p.add_argument('--endpoint',required=True);p.add_argument('--data-root',required=True);p.add_argument('--seed',type=int,required=True)
    args=vars(p.parse_args());jobs=[json.loads(l) for l in Path(args['jobs']).read_text().splitlines()]
    with get_context('spawn').Pool(4,initializer=init,initargs=(args,)) as pool:results=list(pool.map(execute,jobs))
    Path(args['output']).write_text(''.join(json.dumps(r)+'\n' for r in results))
    if any(r['status']!='complete' for r in results):raise SystemExit('Recovery infrastructure error; see output')
if __name__=='__main__':main()
