"""Online PlanCraft HSL, using the shared optimizer, FIFO sampler and exact loss."""
import json,threading,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import train_alfworld_hsl as online
import alfworld_hsl_batch as batch
from alfworld_data_schedule import round_tasks
from plancraft_adapter import DATA,TextSession,context_messages
from plancraft_hindsight import relabel,training_records

HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

class PlanCraftTrainer(online.OnlineHSLTrainer):
    def collect_round(self,step,tasks,rng,version,root):
        started=time.monotonic();directory=root/'rounds'/f'{step:05d}';directory.mkdir(parents=True,exist_ok=True)
        temperature=float(self.config.hsl.get('rollout_temperature',1.))
        if not 0<=temperature<=2:raise ValueError('Invalid primary rollout temperature')
        selected=round_tasks(tasks,int(self.config.hsl.rollouts_per_step),int(self.config.trainer.seed),step,rng,True)
        (directory/'tasks.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in selected))
        recipes=json.loads((DATA/'hindsight_recipes.json').read_text())
        tokenizer=self.model_config.tokenizer;local=threading.local()
        def episode(pair):
            index,e=pair
            if not hasattr(local,'session'):local.session=TextSession()
            s=local.session;s.reset(e,recipes[e['target']]);won=False
            for turn in range(30):
                body=dict(model='qwen3-4b',messages=context_messages(s.messages,tokenizer),temperature=temperature,top_p=1.,max_tokens=128,seed=int(self.config.trainer.seed)+step*100003+index*31+turn)
                request=urllib.request.Request(self.config.hsl.endpoint+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with HTTP.open(request,timeout=180) as response:answer=json.load(response)['choices'][0]['message']['content'] or ''
                _,won=s.step(answer)
                if won:break
            trajectory=list(s.trajectory)
            hindsight=relabel(e,trajectory,recipes,tokenizer,version,s)
            return dict(rollout_id=index,id=e['id'],status='complete',won=won,trajectory=trajectory,hindsight=hindsight)
        with ThreadPoolExecutor(max_workers=int(self.config.hsl.workers)) as pool:rows=list(pool.map(episode,enumerate(selected)))
        (directory/'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return [r['hindsight'] for r in rows],dict(collection_seconds=time.monotonic()-started,rollout_temperature=temperature,rollout_count=len(rows),successes=sum(r['won'] for r in rows),generated_goals=sum(len(r['hindsight']['goals']) for r in rows),relevance_checks=sum(r['hindsight']['relevance_checks'] for r in rows))

if __name__=='__main__':
    batch.training_records=training_records
    online.OnlineHSLTrainer=PlanCraftTrainer
    online.main()
