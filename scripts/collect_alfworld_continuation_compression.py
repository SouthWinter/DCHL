import collect_alfworld_verified_recovery as parent
from alfworld_recovery_compression import compress_result
original_execute=parent.execute

def execute(job):
    result=original_execute(job)
    return compress_result(result,parent.CONFIG['data_root'],max_checks=24)

if __name__=='__main__':
    parent.execute=execute
    parent.main()
