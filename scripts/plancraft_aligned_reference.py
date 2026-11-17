"""Complement a full demonstration with a progress-aligned successful suffix."""
import json,time
from collections import Counter
from plancraft.environment.env import target_and_inventory_to_text_obs
from plancraft_adapter import DATA
from plancraft_hindsight import totals
from plancraft_recovery import choose_jobs as original_jobs,words

def slot_features(inventory):
    return Counter({(int(k),v['type']):v['quantity'] for k,v in inventory.items() if int(k)!=0})

def weighted_jaccard(a,b):
    keys=a.keys()|b.keys();denominator=sum(max(a[k],b[k]) for k in keys)
    return sum(min(a[k],b[k]) for k in keys)/denominator if denominator else 1.

def score(current,reference,target,reference_target):
    a,b=words(target),words(reference_target)
    return (len(a&b)/max(1,len(a|b)),
        .75*weighted_jaccard(totals(current),reference['totals'])+.25*weighted_jaccard(slot_features(current),reference['slots']))

_BANK=None
def state_bank():
    global _BANK
    if _BANK is not None:return _BANK
    _BANK=[]
    for line in (DATA/'demonstration_replay.jsonl').read_text().splitlines():
        trace=json.loads(line);trajectory=trace['trajectory']
        for cut,step in enumerate(trajectory):
            inventory={int(k):v for k,v in step['inventory_before'].items()}
            actions=[s['action'] for s in trajectory[cut:]]
            context=target_and_inventory_to_text_obs(trace['target'],inventory)
            _BANK.append(dict(id=trace['id'],goal=trace['target'],cut=cut,remaining=len(actions),
                totals=totals(inventory),slots=slot_features(inventory),context=context,
                text=context+'\nSuccessful remaining actions from this state:\n'+'\n'.join(actions)))
    return _BANK

def choose_jobs(rows,examples,bank,step,seed):
    started=time.monotonic()
    jobs=original_jobs(rows,examples,bank,step,seed);states=state_bank()
    initial={r['id']:r for r in states if r['cut']==0}
    for job in jobs:
        if job['attempt']==0:continue
        current=job['row']['trajectory'][job['cut']-1]['inventory_after'];target=job['example']['target']
        old=initial[job['reference']['id']];baseline=score(current,old,target,old['goal'])
        available=[r for r in states if r['id']!=job['row']['id'] and r['cut']>0]
        candidate=max(available,key=lambda r:(score(current,r,target,r['goal']),-r['remaining'],r['id'],r['cut']))
        candidate_score=score(current,candidate,target,candidate['goal'])
        job['alignment']=dict(changed=candidate_score>baseline,baseline_score=baseline,candidate_score=candidate_score,reference_cut=candidate['cut'],remaining_actions=candidate['remaining'],original_reference_id=old['id'])
        if candidate_score>baseline:
            job['reference']={k:candidate[k] for k in ('id','goal','context','text')}
            job['reference']['reference_cut']=candidate['cut']
    elapsed=time.monotonic()-started
    for job in jobs:
        if 'alignment' in job:job['alignment']['selection_seconds_per_round']=elapsed
    return jobs
