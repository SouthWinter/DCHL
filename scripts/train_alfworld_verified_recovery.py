import json,os,subprocess,time
from pathlib import Path
import train_alfworld_hsl as online
from alfworld_goal_value_loop import BufferValueController
from alfworld_verified_recovery import VerifiedRecoveryBuffer,mixed_records,choose_jobs,recovery_entry

class RecoveryTrainer(online.OnlineHSLTrainer):
    collector_script='collect_alfworld_verified_recovery.py'
    def collect_round(self,step,tasks,rng,version,root):
        relabeled,metrics=super().collect_round(step,tasks,rng,version,root)
        directory=root/'rounds'/f'{step:05d}'
        raw=[json.loads(l) for l in (directory/'rollouts.jsonl').read_text().splitlines()]
        raw=list({r['rollout_id']:r for r in raw}.values())
        bank=json.loads((root.parent/'recovery_reference_bank.json').read_text())
        jobs=choose_jobs(raw,bank,step,int(self.config.trainer.seed))
        path=directory/'recovery_jobs.jsonl';path.write_text(''.join(json.dumps(j)+'\n' for j in jobs))
        output=directory/'recovery_results.jsonl';started=time.monotonic()
        if jobs:
            command=[str(self.config.hsl.environment_python),'-u',str(Path(__file__).with_name(self.collector_script)),
                '--jobs',str(path),'--output',str(output),'--endpoint',self.config.hsl.endpoint,
                '--data-root',self.config.hsl.data_root,'--seed',str(int(self.config.trainer.seed))]
            with (directory/'recovery_collector.log').open('a') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1'),check=True)
            results=[json.loads(l) for l in output.read_text().splitlines()]
        else:results=[];output.write_text('')
        unique={}
        for result in sorted(results,key=lambda r:r['attempt']):
            entry=recovery_entry(result)
            if entry:unique.setdefault(entry['key'],entry)
        relabeled[0]['_verified_recoveries']=list(unique.values())
        metrics.update(recovery_seconds=time.monotonic()-started,recovery_attempts=len(results),
            recovery_successes=sum(r['won'] for r in results),recovery_unique=len(unique),
            recovery_generated_steps=sum(r['generated_steps'] for r in results),
            recovery_replayed_steps=sum(r['replayed_steps'] for r in results))
        metrics.update(recovery_compression_steps=sum(r.get('compression_steps',0) for r in results),
            recovery_compression_removed=sum(r.get('compression_removed',0) for r in results))
        return relabeled,metrics

if __name__=='__main__':
    online.HSLBuffer=VerifiedRecoveryBuffer;online.mixed_records=mixed_records
    online.ODMController=BufferValueController;online.OnlineHSLTrainer=RecoveryTrainer
    online.main()
