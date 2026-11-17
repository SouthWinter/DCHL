"""SFT on original-goal verified continuations of learner prefixes."""
import copy,json,random,re
from collections import Counter
from alfworld_stable_evidence import StableEvidenceBuffer
from alfworld_evidence_sampling import mixed_records as base_records
from alfworld_hsl_buffer import training_records

def query_words(text):return set(re.findall(r'[a-z]+',text.lower()))

def retrieve_reference(row,cut,bank):
    goal=query_words(row['task_description'])
    observation=query_words(row['trajectory'][cut-1]['result_observation'])
    def score(entry):
        a=query_words(entry['goal']);b=query_words(entry['context'])
        return (len(goal&a)/max(1,len(goal|a))+.25*len(observation&b)/max(1,len(observation|b)),entry['id'])
    return max(bank,key=score)

def choose_jobs(rows,bank,step,seed):
    failed=sorted((r for r in rows if not r['won'] and len(r['trajectory'])>1),key=lambda r:r['rollout_id'])
    rng=random.Random(seed+step*8191)
    selected=rng.sample(failed,min(4,len(failed)))
    jobs=[]
    for row in selected:
        cut=min(24,max(1,len(row['trajectory'])//2))
        # Prefer a prefix immediately before the first observable invalid action.
        for i,s in enumerate(row['trajectory'][:cut+1]):
            if i>0 and s['result_observation'].startswith('Nothing happens'):
                cut=i;break
        ref=retrieve_reference(row,cut,bank)
        for attempt in range(2):jobs.append(dict(row=row,cut=cut,reference=ref,attempt=attempt,round=step))
    return jobs

def recovery_entry(result):
    if result.get('status')!='complete' or not result.get('won'):return None
    cut=result['cut'];steps=result['trajectory']
    assert 0<cut<len(steps) and result['original_goal_verified']
    rollout=dict(gamefile=result['gamefile'],policy_version=result['policy_version'],
        rollout_id=result['rollout_id'],steps=steps,dialogue_hsl=True,response_contract='action_only_v1')
    goal=dict(instruction=result['task_description'],task_type=result['task_type'],prefix_length=len(steps),
              action_loss_mask=[i>=cut for i in range(len(steps))])
    return dict(rollout=rollout,goal=goal,round=result['round'],cut=cut,
                key=f"{result['round']}:{result['rollout_id']}:{cut}")

class VerifiedRecoveryBuffer(StableEvidenceBuffer):
    feedback_metric_key='verified_recovery'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.recoveries=[];self.recovery_demand=[]
    def append_round(self,results,policy_version):
        extra=[x for r in results for x in r.get('_verified_recoveries',[])]
        clean=[{k:v for k,v in r.items() if k!='_verified_recoveries'} for r in results]
        super().append_round(clean,policy_version)
        self.recovery_demand.extend(dict(round=self.round_index,task=r['requested_task_type'],failed=not r['source_success']) for r in clean)
        self.recovery_demand=[r for r in self.recovery_demand if self.round_index-r['round']<10]
        pool={r['key']:r for r in self.recoveries}
        for r in extra:pool.setdefault(r['key'],r)
        self.recoveries=sorted((r for r in pool.values() if self.round_index-r['round']<10),key=lambda r:(r['round'],r['key']))[-128:]
    def state_dict(self):return dict(super().state_dict(),recovery_bank=self.recoveries,recovery_demand=self.recovery_demand)
    @classmethod
    def from_state_dict(cls,state):
        b=super().from_state_dict(state);b.recoveries=state.get('recovery_bank',[]);b.recovery_demand=state.get('recovery_demand',[]);return b

def mixed_records(demonstrations,buffer,rng,**kwargs):
    rows,provenance=base_records(demonstrations,buffer,rng,**kwargs)
    if not buffer.recoveries:return rows,provenance
    attempts=Counter(r['task'] for r in buffer.recovery_demand)
    failures=Counter(r['task'] for r in buffer.recovery_demand if r['failed'])
    groups={}
    for e in buffer.recoveries:groups.setdefault(e['goal']['task_type'],[]).append(e)
    aux=random.Random(991003+buffer.round_index);keys=sorted(groups)
    chosen=[]
    for task in aux.choices(keys,weights=[(failures[k]+1)/(attempts[k]+4) for k in keys],k=4):
        chosen.append(aux.choice(groups[task]))
    for e in chosen:
        records=[r for r in training_records(e['rollout'],e['goal']) if r['action_supervised']]
        for r in records:r.update(trajectory_action_coefficient=.05/len(chosen)/len(records),sampling_draw_id=-1)
        rows.extend(records);provenance.append(dict(source='verified_recovery',key=e['key'],cut=e['cut'],actions=len(records),coefficient=.05/len(chosen)))
    return rows,provenance
