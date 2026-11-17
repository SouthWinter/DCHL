"""Freeze recipe inputs and tokenize the exact action-wise online SFT records."""
import json,random
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from plancraft.environment.recipes import RECIPES
from plancraft.environment.search import gold_search_recipe
from plancraft_adapter import DATA
from plancraft_hindsight import action_records,relabel

if __name__=='__main__':
    random.seed(42);np.random.seed(42)
    recipes=json.loads((DATA/'recipes.json').read_text())
    for target in sorted(RECIPES):
        if target not in recipes:recipes[target]=gold_search_recipe(target)
    (DATA/'hindsight_recipes.json').write_text(json.dumps(recipes,indent=2)+'\n')
    tokenizer=AutoTokenizer.from_pretrained(__import__('os').environ['DCHL_BASE_MODEL'],local_files_only=True)
    output=[]
    for demo in pq.read_table(DATA/'train.parquet').to_pylist():
        records=action_records(demo['messages'],tokenizer)
        for index,row in enumerate(records):row.update(source_episode_id=demo['source_episode_id'],step_index=index+1)
        output.extend(records)
    pd.DataFrame(output).to_parquet(DATA/'hsl_demonstrations.parquet',index=False)
    examples={e['id']:e for e in map(json.loads,(DATA/'train.jsonl').read_text().splitlines())}
    traces=list(map(json.loads,(DATA/'demonstration_replay.jsonl').read_text().splitlines()))
    # Select expert traces spanning horizon lengths, including the longest.
    traces=sorted(traces,key=lambda x:x['actions']);selected=[traces[i] for i in np.linspace(0,len(traces)-1,12,dtype=int)]
    labels=[relabel(examples[t['id']],t['trajectory'],recipes,tokenizer,'validation') for t in selected]
    assert all(x['goals'] for x in labels)
    report=dict(status='PASS',original_trajectories=945,action_records=len(output),replayed_trajectories=len(labels),goals=sum(len(x['goals']) for x in labels),relevance_checks=sum(x['relevance_checks'] for x in labels))
    (DATA/'hsl_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
