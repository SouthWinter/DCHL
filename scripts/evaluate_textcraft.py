"""Native behavior evaluation; TRAIN supply probes are explicitly labeled."""
import argparse
import json
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from transformers import AutoTokenizer
from textcraft_adapter import DATA,TextSession,context_messages,MAX_STEPS

HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--examples',required=True)
    p.add_argument('--endpoints',required=True);p.add_argument('--run',required=True);p.add_argument('--temperature',type=float,default=0.)
    p.add_argument('--scope',required=True);p.add_argument('--workers',type=int,default=48);a=p.parse_args()
    run=Path(a.run);output=run/'outputs';output.mkdir(parents=True,exist_ok=True)
    tasks=[json.loads(l) for l in Path(a.examples).read_text().splitlines()]
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    tokenizer.chat_template=(DATA/'fixed_chat_template.jinja').read_text()
    endpoints=a.endpoints.split(',');local=threading.local();sessions=[];started=time.monotonic()
    def episode(index,e):
        dest=output/f"{e['id']}.json"
        if dest.exists():
            r=json.loads(dest.read_text())
            if r['status']=='complete':return r
        if not hasattr(local,'session'):local.session=TextSession();sessions.append(local.session)
        s=local.session
        try:
            s.reset(e)
            for turn in range(MAX_STEPS):
                body=dict(model='textcraft-agent',messages=context_messages(s.messages,tokenizer),
                    temperature=a.temperature,top_p=1.,max_tokens=128,seed=42+index*101+turn)
                req=urllib.request.Request(endpoints[index%len(endpoints)]+'/v1/chat/completions',
                    data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with HTTP.open(req,timeout=180) as response:answer=json.load(response)['choices'][0]['message']['content'] or ''
                s.step(answer)
                if s.done:break
            r=dict(id=e['id'],goal=e['goal'],depth=e['depth'],status='complete',won=s.won,
                score=s.score,steps=len(s.trajectory),trajectory=s.trajectory)
        except Exception as exc:r=dict(id=e['id'],status='infrastructure_error',error=repr(exc))
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False));tmp.replace(dest);return r
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            results=[]
            for future in as_completed([pool.submit(episode,i,e) for i,e in enumerate(tasks)]):
                results.append(future.result())
                if len(results)%50==0:print('episodes',len(results),'successful',sum(r.get('won',False) for r in results),flush=True)
    finally:
        for s in sessions:s.close()
    assert {r['id'] for r in results}=={e['id'] for e in tasks}
    groups=defaultdict(list)
    for r in results:
        if r['status']=='complete':groups[r['depth']].append(r)
    report=dict(status='PASS' if all(r['status']=='complete' for r in results) else 'FAILED',scope=a.scope,
        episodes=len(results),infrastructure_errors=sum(r['status']!='complete' for r in results),
        successes=sum(r.get('won',False) for r in results),temperature=a.temperature,seed=42,
        max_steps=MAX_STEPS,max_tokens_per_action=128,wall_seconds=time.monotonic()-started,
        by_depth={k:dict(episodes=len(v),successes=sum(r['won'] for r in v)) for k,v in groups.items()})
    (run/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report),flush=True)
    assert report['status']=='PASS'

if __name__=='__main__':main()
