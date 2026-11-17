"""Prepare complete ETO conversations with trajectory batches and zero-loss padding."""
import copy
import json
import random
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/alfworld_eto_sft800_v1'

def main():
    source=json.loads((OUT/'source.json').read_text())
    assert len({r['id'] for r in source})==len(source)
    selected=random.Random(42).sample(sorted(source,key=lambda r:r['id']),800)
    tokenizer=AutoTokenizer.from_pretrained(__import__('os').environ['DCHL_BASE_MODEL'],local_files_only=True)
    rows=[];lengths=[]
    for r in selected:
        messages=[dict(role={'human':'user','gpt':'assistant'}[m['from']],content=m['value'],loss_on_this_turn=m['from']=='gpt') for m in r['conversations']]
        assert all(m['role']==('user' if i%2==0 else 'assistant') for i,m in enumerate(messages))
        lengths.append(len(tokenizer.apply_chat_template(messages,tokenize=True)))
        rows.append(dict(record_id=r['id'],messages=messages,enable_thinking=False))
    pq.write_table(pa.Table.from_pylist(rows),OUT/'demonstrations.parquet')
    padding=copy.deepcopy(min(rows,key=lambda r:len(str(r))))
    for m in padding['messages']:m['loss_on_this_turn']=False
    schedule=[]
    for epoch in range(3):
        cycle=rows.copy();random.Random(42+epoch).shuffle(cycle)
        for i in range(0,800,32):schedule.extend(cycle[i:i+32]+[copy.deepcopy(padding) for _ in range(3)])
    pq.write_table(pa.Table.from_pylist(schedule),OUT/'train.parquet')
    (OUT/'instruction.json').write_text(json.dumps(selected[0]['conversations'][:2],ensure_ascii=False))
    assert all(r['conversations'][:2]==selected[0]['conversations'][:2] for r in selected)
    report=dict(source='https://huggingface.co/datasets/agent-eto/eto-sft-trajectory/resolve/main/data/alfworld_sft.json',source_rows=len(source),selected_ids=[r['id'] for r in selected],unique_trajectories=800,updates=75,effective_batch=32,physical_batch=35,epochs=3,max_tokens=max(lengths),over4096=sum(n>4096 for n in lengths),truncation='right at4096, as ETO',supervised='all assistant turns including OK, thoughts, actions',padding='3 zero-loss rows per update')
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!='selected_ids'})

if __name__=='__main__':main()
