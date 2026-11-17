"""Transfer the stable evidence + long generation + compressed supervision recipe."""
import json,time
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
import train_plancraft_hsl as base
import alfworld_verified_recovery as recovery
from alfworld_goal_value_loop import BufferValueController
from plancraft_adapter import DATA
from plancraft_hindsight import training_records
from plancraft_recovery import evidence_labels,choose_jobs,execute,make_bank

class ContinuationTrainer(base.PlanCraftTrainer):
    def collect_round(self,step,tasks,rng,version,root):
        _,metrics=super().collect_round(step,tasks,rng,version,root)
        directory=root/'rounds'/f'{step:05d}'
        rows=[json.loads(l) for l in (directory/'rollouts.jsonl').read_text().splitlines()]
        examples={e['id']:e for e in tasks};recipes=json.loads((DATA/'hindsight_recipes.json').read_text())
        if not hasattr(self,'reference_bank'):
            self.reference_bank=make_bank(pq.read_table(DATA/'train.parquet').to_pylist(),examples)
        labels=[evidence_labels(r) for r in rows]
        jobs=choose_jobs(rows,examples,self.reference_bank,step,int(self.config.trainer.seed))
        (directory/'recovery_jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))
        started=time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda j:execute(j,recipes,self.model_config.tokenizer,self.config.hsl.endpoint),jobs))
        (directory/'recoveries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
        for label,row in zip(labels,rows):label['_verified_recoveries']=[r['entry'] for r in results if r['rollout_id']==row['rollout_id'] and r.get('entry')]
        metrics.update(recovery_seconds=time.monotonic()-started,recovery_attempts=len(results),recovery_successes=sum(r['won'] for r in results),recovery_generated_steps=sum(r['generated_steps'] for r in results),recovery_compression_checks=sum(r.get('compression_checks',0) for r in results),recovery_removed_actions=sum(r.get('removed_actions',0) for r in results))
        return labels,metrics

if __name__=='__main__':
    base.batch.training_records=training_records
    recovery.training_records=training_records
    online=base.online;online.HSLBuffer=recovery.VerifiedRecoveryBuffer
    online.mixed_records=recovery.mixed_records;online.ODMController=BufferValueController
    online.OnlineHSLTrainer=ContinuationTrainer
    online.main()
