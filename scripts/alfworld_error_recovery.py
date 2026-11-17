"""Post-error recovery with public execution-evidence suffix supervision."""
import random
from alfworld_verified_recovery import (VerifiedRecoveryBuffer,retrieve_reference,
    recovery_entry as full_suffix_entry,mixed_records)
from alfworld_execution_evidence import execution_event

def explicit_failure(step):
    return (step['action']=='invalid action' or
        step['result_observation'].lower().startswith(('nothing happens','nothing happened')))

def error_cut(steps):
    return next((i+1 for i,s in enumerate(steps[:24]) if explicit_failure(s)),None)

def choose_jobs(rows,bank,step,seed):
    eligible=sorted((r for r in rows if not r['won'] and error_cut(r['trajectory']) is not None),key=lambda r:r['rollout_id'])
    rng=random.Random(seed+step*8191);selected=rng.sample(eligible,min(4,len(eligible)))
    jobs=[]
    for row in selected:
        cut=error_cut(row['trajectory']);ref=retrieve_reference(row,cut,bank)
        for attempt in range(2):jobs.append(dict(row=row,cut=cut,reference=ref,attempt=attempt,round=step))
    return jobs

def recovery_entry(result):
    entry=full_suffix_entry(result)
    if entry is None:return None
    cut=entry['cut'];steps=entry['rollout']['steps']
    assert explicit_failure(steps[cut-1]),'Recovery prefix must end after an observed error'
    # Keep every observed history turn; only successful tool receipts become targets.
    mask=[i>=cut and execution_event(s) is not None for i,s in enumerate(steps)]
    if not any(mask):return None
    entry['goal']['action_loss_mask']=mask
    entry['supervised_actions']=sum(mask)
    entry['masked_suffix_actions']=len(steps)-cut-sum(mask)
    return entry

class ErrorRecoveryBuffer(VerifiedRecoveryBuffer):
    feedback_metric_key='error_recovery'
    def append_round(self,results,policy_version):
        super().append_round(results,policy_version)
        # Align auxiliary demand with the same observed-error condition used to
        # construct data. Denominator still includes all original task attempts.
        for demand,row in zip(self.recovery_demand[-len(results):],results,strict=True):
            demand['failed']=not row['source_success'] and error_cut(row['steps']) is not None
