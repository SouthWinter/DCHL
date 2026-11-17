"""Independent greedy PlanCraft episodes through local OpenAI-compatible servers."""
import argparse,hashlib,json,threading,time,urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from transformers import AutoTokenizer
from plancraft_adapter import DATA,TextSession,context_messages
LOCAL_HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

DATE_STRING=None


def request_action(endpoint,messages):
    body=dict(model='plancraft-agent',messages=messages,temperature=0,max_tokens=128,seed=42)
    if DATE_STRING is not None:
        body['chat_template_kwargs']={'date_string':DATE_STRING}
    req=urllib.request.Request(endpoint+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with LOCAL_HTTP.open(req,timeout=180) as r:data=json.load(r)
    return data['choices'][0]['message']['content'] or ''

def evaluate(examples,recipes,tokenizer,endpoints,output,workers=20,examples_file=None):
    output.mkdir(parents=True,exist_ok=True);local=threading.local();started=time.monotonic()
    def episode(index,e):
        dest=output/(e['id']+'.json')
        if dest.exists():
            old=json.loads(dest.read_text())
            if old.get('status')=='complete':return old
        try:
            if not hasattr(local,'session'):local.session=TextSession()
            session=local.session;session.reset(e,recipes[e['target']]);won=False
            for _ in range(30):
                prompt=context_messages(session.messages,tokenizer)
                response=request_action(endpoints[index%len(endpoints)],prompt)
                _,won=session.step(response)
                if won:break
            result=dict(id=e['id'],status='complete',won=won,steps=session.steps,target=e['target'],complexity=e['complexity_split'],trajectory=session.trajectory,invalid_actions=sum(not r['valid'] for r in session.trajectory))
        except Exception as exc:result=dict(id=e['id'],status='infrastructure_error',error=f'{type(exc).__name__}: {exc}')
        dest.write_text(json.dumps(result)+'\n');return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(episode,i,e) for i,e in enumerate(examples)]
        results=[]
        for f in as_completed(futures):
            results.append(f.result())
            if len(results)%40==0:print('evaluated',len(results),'successes',sum(r.get('won',False) for r in results),flush=True)
    errors=[r for r in results if r['status']!='complete']
    report=dict(status='PASS' if not errors else 'FAILED',episodes=len(results),successes=sum(r.get('won',False) for r in results),infrastructure_errors=len(errors),success_rate=sum(r.get('won',False) for r in results)/len(results),wall_seconds=time.monotonic()-started,successes_by_complexity=dict(Counter(r['complexity'] for r in results if r.get('won'))),max_steps=30,decoding='greedy',recipe_information='Frozen public target recipes; no planner or optimal actions in evaluation')
    if DATE_STRING is not None:report['fixed_date_string']=DATE_STRING
    if examples_file is not None:
        report['examples_file']=str(examples_file)
        report['examples_sha256']=hashlib.sha256(examples_file.read_bytes()).hexdigest()
    (output.parent/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    if errors:raise RuntimeError(f'{len(errors)} episode infrastructure errors')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--endpoints',required=True);p.add_argument('--output',required=True);p.add_argument('--split',default='test',choices=['val','test']);p.add_argument('--workers',type=int,default=20)
    p.add_argument('--tokenizer',default=__import__('os').environ['DCHL_BASE_MODEL'])
    p.add_argument('--date-string',default=None)
    p.add_argument('--examples-file',default=None)
    a=p.parse_args();DATE_STRING=a.date_string
    examples_file=Path(a.examples_file) if a.examples_file else DATA/f'{a.split}.jsonl'
    examples=[json.loads(l) for l in examples_file.read_text().splitlines()]
    recipes=json.loads((DATA/'recipes.json').read_text());tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    if DATE_STRING is not None:
        original=tokenizer.chat_template
        tokenizer.chat_template=original.replace('strftime_now("%d %b %Y")',repr(DATE_STRING))
        assert 'strftime_now(' not in tokenizer.chat_template
    print(json.dumps(evaluate(examples,recipes,tokenizer,a.endpoints.split(','),Path(a.output),a.workers,examples_file),indent=2))
