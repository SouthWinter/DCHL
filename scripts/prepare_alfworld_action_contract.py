"""Create consistent full-dialogue SFT and per-action HSL data from the same800 demos."""
import copy
import json
import random
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from alfworld_action_contract import canonical_messages

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'data/alfworld_eto_sft800_v1/demonstrations.parquet'
    out=ROOT/'data/alfworld_action_contract_v1'
    out.mkdir(exist_ok=True)
    full=pq.read_table(source).to_pylist()
    rows=[]
    for demo in full:
        demo['messages']=canonical_messages(demo['messages'])
        step=0
        for i,m in enumerate(demo['messages'][2:],start=2):
            if m['role']!='assistant':continue
            step+=1
            messages=copy.deepcopy(demo['messages'][:i+1])
            for msg in messages:msg['loss_on_this_turn']=False
            messages[-1]['loss_on_this_turn']=True
            rows.append(dict(record_id=f"{demo['record_id']}#{step}",source_episode_id=demo['record_id'],step_index=step,messages=messages,enable_thinking=False))
    pq.write_table(pa.Table.from_pylist(full),out/'sft_demonstrations.parquet')
    padding=copy.deepcopy(min(full,key=lambda r:len(str(r))))
    for message in padding['messages']:message['loss_on_this_turn']=False
    schedule=[]
    for epoch in range(3):
        cycle=full.copy();random.Random(42+epoch).shuffle(cycle)
        for i in range(0,len(cycle),32):
            schedule.extend(cycle[i:i+32]+[copy.deepcopy(padding) for _ in range(3)])
    pq.write_table(pa.Table.from_pylist(schedule),out/'train.parquet')
    pq.write_table(pa.Table.from_pylist(rows),out/'hsl_demonstrations.parquet')
    report=dict(trajectories=len(full),action_records=len(rows),source=str(source),response_contract='action_only_v1',requires_new_sft=True,training_status='NOT_STARTED')
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)

if __name__=='__main__':main()
