"""Download public demonstration sources (no model weights)."""
import argparse,json,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES={
 'alfworld':{'data/alfworld_eto_sft800_v1/source.json':'https://huggingface.co/datasets/agent-eto/eto-sft-trajectory/resolve/main/data/alfworld_sft.json'},
 'babyai':{'data/babyai_screen_v1/source/babyai_train.json':'https://huggingface.co/datasets/AgentGym/AgentTraj-L/resolve/main/babyai_train.json','data/babyai_screen_v1/source/babyai_test.json':'https://huggingface.co/datasets/AgentGym/AgentEval/resolve/main/babyai_test.json'}}
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',choices=FILES);a=p.parse_args();manifest=[]
 for path,url in FILES[a.dataset].items():
  dst=ROOT/path;dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():
   content=urllib.request.urlopen(url,timeout=120).read();json.loads(content);tmp=dst.with_suffix('.download');tmp.write_bytes(content);tmp.replace(dst)
  manifest.append(dict(path=path,url=url,sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
