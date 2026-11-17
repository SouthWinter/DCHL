"""Build the TRAIN collection panel from public native ALFWorld tasks."""
import json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    data=Path(os.environ['ALFWORLD_DATA']).resolve()
    src=ROOT/'external/alfworld_text_data/data/train_games.jsonl'
    rows=[json.loads(x)['metadata'] for x in src.read_text().splitlines()]
    tasks={str(data/r['gamefile'].split('${ALFWORLD_DATA}/',1)[1]):r['task_type'] for r in rows}
    assert all(Path(p).is_file() for p in tasks), 'Extract ALFWorld assets before preparing the task pool'
    out=ROOT/'data/alfworld_dialogue_hsl_v1';out.mkdir(parents=True,exist_ok=True)
    (out/'tasks.jsonl').write_text(''.join(json.dumps(dict(gamefile=p,task_type=tasks[p]))+'\n' for p in sorted(tasks)))
if __name__=='__main__':main()
