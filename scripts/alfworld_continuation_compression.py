"""Extend the successful243 recipe's recovery horizon without changing proposals."""
from alfworld_verified_recovery import choose_jobs as original_jobs
from alfworld_expanded_recovery import evidence_entry

def choose_jobs(rows,bank,step,seed):
    jobs=original_jobs(rows,bank,step,seed)
    for job in jobs:job['max_steps']=24
    return jobs
