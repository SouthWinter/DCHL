"""PlanCraft adapters for stable execution evidence and verified long recovery."""
import copy,json,random,re,urllib.request
from concurrent.futures import ThreadPoolExecutor
from plancraft_adapter import TextSession,context_messages,replay_actions,compress_success
from plancraft_hindsight import totals,action_records

HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def evidence_labels(row):
    result=copy.deepcopy(row['hindsight'])
    result.update(rollout_id=row['rollout_id'],requested_task_type='craft',source_success=row['won'])
    for goal in result['goals']:
        proof=[]
        for i,step in enumerate(row['trajectory'][:goal['prefix_length']]):
            before=totals(step['inventory_before']);after=totals(step['inventory_after'])
            if goal['action_loss_mask'][i] and any(n>before[k] for k,n in after.items()):proof.append(i)
        if not proof:raise ValueError('Goal has no supervised manufacturing evidence')
        goal.update(object_type=goal['instruction'],feedback_steps=proof)
        for i,record in enumerate(goal['records']):record['step']=i
    return result

def words(text):return set(re.findall(r'[a-z]+',text.lower()))

def choose_jobs(rows,examples,bank,step,seed):
    failed=sorted((r for r in rows if not r['won'] and len(r['trajectory'])>1),key=lambda r:r['rollout_id'])
    selected=random.Random(seed+step*8191).sample(failed,min(4,len(failed)));jobs=[]
    for row in selected:
        cut=min(24,max(1,len(row['trajectory'])//2))
        for i,s in enumerate(row['trajectory'][:cut+1]):
            if i>0 and (not s['valid'] or not s['changed']):cut=i;break
        example=examples[row['id']];goal=words(example['target']);obs=words(row['trajectory'][cut-1]['observation'])
        def score(ref):
            a=words(ref['goal']);b=words(ref['context'])
            return len(goal&a)/max(1,len(goal|a))+.25*len(obs&b)/max(1,len(obs|b)),ref['id']
        reference=max((r for r in bank if r['id']!=row['id']),key=score)
        for attempt in range(2):jobs.append(dict(row=row,example=example,cut=cut,reference=reference,attempt=attempt,round=step,seed=seed))
    return jobs

def execute(job,recipes,tokenizer,endpoint,forced_actions=None):
    s=TextSession();row=job['row'];e=job['example'];cut=job['cut'];recipe=recipes[e['target']]
    s.reset(e,recipe)
    for original in row['trajectory'][:cut]:
        _,won=s.step(original['action']);actual=s.trajectory[-1]
        for field in ('action','inventory_before','inventory_after','observation'):
            a,b=actual[field],original[field]
            if field.startswith('inventory_'):a={int(k):v for k,v in a.items()};b={int(k):v for k,v in b.items()}
            if a!=b:raise ValueError('Recovery prefix mismatch: '+field)
        if won:raise ValueError('Recovery prefix already complete')
    hint='\nA successful example from another training episode follows. Adapt its actions to the current inventory and task; slot identifiers may differ.\n'+job['reference']['text']
    generated=0;won=False
    if job.get('bridge_action'):
        _,won=s.step(job['bridge_action']);generated=1
        if not s.trajectory[-1]['changed']:raise ValueError('Verified bridge no longer changes the state')
    for j in range(24-generated):
        if won:break
        if forced_actions is not None:
            if j>=len(forced_actions):break
            answer=forced_actions[j]
        else:
            guided=copy.deepcopy(s.messages);guided[0]['content']+=hint
            body=dict(model='qwen3-4b',messages=context_messages(guided,tokenizer),temperature=1.,top_p=1.,max_tokens=128,seed=job['seed']+job['round']*100003+row['rollout_id']*1009+job['attempt']*503+j)
            req=urllib.request.Request(endpoint+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with HTTP.open(req,timeout=180) as response:answer=json.load(response)['choices'][0]['message']['content'] or ''
        _,won=s.step(answer);generated+=1
        if won:break
    result=dict(status='complete',won=won,round=job['round'],rollout_id=row['rollout_id'],source_id=row['id'],cut=cut,attempt=job['attempt'],reference_id=job['reference']['id'],generated_steps=generated,trajectory=copy.deepcopy(s.trajectory))
    if job.get('bridge_action'):result.update(bridge_action=job['bridge_action'],bridge=job['bridge'])
    if won:
        compressed=compress_success(s,e,recipe,[x['action'] for x in s.trajectory],cut,max_checks=24)
        records=action_records(compressed['messages'][:-1],tokenizer)
        mask=[i>=cut and t['valid'] and t['changed'] for i,t in enumerate(compressed['trajectory'])]
        if not any(mask):raise ValueError('Successful recovery has no effective suffix')
        entry=dict(key=f"{job['round']}:{row['rollout_id']}:{cut}",round=job['round'],cut=cut,
            rollout=dict(gamefile=row['id'],rollout_id=row['rollout_id'],policy_version=row['hindsight']['policy_version']),
            goal=dict(instruction=e['target'],task_type='craft',prefix_length=len(records),action_loss_mask=mask,records=records))
        result.update(entry=entry,compressed_trajectory=compressed['trajectory'],compression_checks=compressed['checks'],removed_actions=compressed['removed'])
    return result

def make_bank(demos,examples):
    bank=[]
    for demo in demos:
        identifier=demo['source_episode_id'];messages=demo['messages']
        actions=[m['content'] for m in messages if m['role']=='assistant']
        context=messages[1]['content'].split('\n\n')[0]
        bank.append(dict(id=identifier,goal=examples[identifier]['target'],context=context,
            text=context+'\nSuccessful actions:\n'+'\n'.join(actions)))
    return bank
