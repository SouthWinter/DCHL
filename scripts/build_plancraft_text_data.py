"""Re-execute official expert demonstrations and freeze public target recipes."""
import json,random,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from plancraft.simple import get_plancraft_examples
from plancraft.environment.search import gold_search_recipe
from plancraft_adapter import ROOT,DATA,TextSession

if __name__=='__main__':
    DATA.mkdir(exist_ok=True)
    splits={s:[e for e in get_plancraft_examples(s) if not e.impossible] for s in ('train','val','test')}
    random.seed(42);np.random.seed(42)
    recipes={target:gold_search_recipe(target) for target in sorted({e.target for es in splits.values() for e in es})}
    (DATA/'recipes.json').write_text(json.dumps(recipes,indent=2)+'\n')
    for split,examples in splits.items():
        (DATA/f'{split}.jsonl').write_text(''.join(e.model_dump_json()+'\n' for e in examples))
    ids=[{e.id for e in es} for es in splits.values()]
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    tokenizer=AutoTokenizer.from_pretrained(__import__('os').environ['DCHL_BASE_MODEL'],local_files_only=True)
    session=TextSession();rows=[];lengths=[];failures=[];observed=[]
    for e in splits['train']:
        source=ROOT/f'external/plancraft/oracle_trajectories/train/oa/{e.id}.json'
        demo=json.loads(source.read_text());session.reset(e,recipes[e.target]);won=False
        for m in demo['messages']:
            if m['role']=='assistant':
                _,won=session.step(m['content'])
                if won:break
        if not won:failures.append(e.id);continue
        messages=session.messages[:-1]
        for m in messages:m['loss_on_this_turn']=m['role']=='assistant'
        length=len(tokenizer.apply_chat_template(messages,tokenize=True));lengths.append(length)
        rows.append(dict(messages=messages,enable_thinking=False,source_episode_id=e.id,task_type=e.recipe_type,actions=session.steps))
        observed.append(dict(id=e.id,target=e.target,actions=session.steps,won=won,trajectory=session.trajectory))
        if len(rows)%150==0:print('verified',len(rows),flush=True)
    (DATA/'demonstration_replay.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in observed))
    report=dict(status='PASS' if not failures and max(lengths)<=12288 else 'FAILED',source='https://github.com/gautierdag/plancraft',package_version='0.4.9',source_commit_date=subprocess.check_output(['git','-C',str(ROOT/'external/plancraft'),'log','-1','--format=%cI'],text=True).strip(),splits={s:len(es) for s,es in splits.items()},demonstrations=len(rows),failed_replays=failures,tokens=sum(lengths),max_tokens=max(lengths),max_actions=max(r['actions'] for r in rows),over30_actions=sum(r['actions']>30 for r in rows),recipe_information='Official gold_search_recipe(target), frozen once with seed42; public target recipes only, no optimal path or inventory-specific plan in policy input',supervision='All expert move/smelt actions; observations rebuilt by current official environment; full independent dialogues',source_actions='Official train/oa demonstrations; no new teacher generation')
    (DATA/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['status']=='PASS',report
    pd.DataFrame(rows).to_parquet(DATA/'train.parquet',index=False)
    print(json.dumps(report,indent=2))
