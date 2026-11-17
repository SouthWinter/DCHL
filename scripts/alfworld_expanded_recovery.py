"""Broader recovery search with alternate prefixes and distinct references."""
import random
from alfworld_verified_recovery import (VerifiedRecoveryBuffer,retrieve_reference,
    recovery_entry as full_suffix_entry,mixed_records)
from alfworld_error_recovery import error_cut
from alfworld_execution_evidence import execution_event

def evidence_entry(result):
    entry=full_suffix_entry(result)
    if entry is None:return None
    entry['goal']['action_loss_mask']=[i>=entry['cut'] and execution_event(s) is not None
        for i,s in enumerate(entry['rollout']['steps'])]
    return entry if any(entry['goal']['action_loss_mask']) else None

def choose_jobs(rows,bank,step,seed):
    rng=random.Random(seed+step*8191)
    failed=sorted((r for r in rows if not r['won'] and len(r['trajectory'])>2),key=lambda r:r['rollout_id'])
    selected=rng.sample(failed,min(4,len(failed)));jobs=[]
    for row in selected:
        n=len(row['trajectory'])
        first=error_cut(row['trajectory']) or max(1,min(24,n//3))
        second=max(1,min(24,2*n//3))
        if second==first:second=max(1,first-1) if first>1 else min(24,n-1)
        ref1=retrieve_reference(row,first,bank)
        ref2=retrieve_reference(row,second,[r for r in bank if r['id']!=ref1['id']])
        for attempt,(cut,reference) in enumerate(((first,ref1),(second,ref2))):
            jobs.append(dict(row=row,cut=cut,reference=reference,attempt=attempt,round=step,max_steps=24))
    return jobs

class ExpandedRecoveryBuffer(VerifiedRecoveryBuffer):
    def append_round(self,results,policy_version):
        # Different prefixes of the same source share one exposure unit: keep
        # the one requiring more real learner history, when both recover.
        clean=[]
        for row in results:
            row=dict(row);groups={}
            for e in row.get('_verified_recoveries',[]):
                key=(e['round'],e['rollout']['gamefile'],e['rollout']['rollout_id'])
                if key not in groups or e['cut']>groups[key]['cut']:groups[key]=e
            if '_verified_recoveries' in row:row['_verified_recoveries']=list(groups.values())
            clean.append(row)
        super().append_round(clean,policy_version)
