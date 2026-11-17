"""Fetch pinned upstream code; apply the released supervised-training patch."""
import argparse,subprocess,shutil,json,urllib.request,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPOS={
 'verl':('https://github.com/volcengine/verl.git','bec9ef74768dd201881cd4e54cd0385e87caae27'),
 'plancraft':('https://github.com/gautierdag/plancraft.git','7617e9ff0c68e3158e3ba8ea260e6d83c454ba6f'),
 'AgentGym':('https://github.com/WooooDyy/AgentGym.git','3ef9235d23e68e7c2920c5422ad957dc8ced5c6c')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--only',choices=list(REPOS));a=p.parse_args()
 for name,(url,sha) in REPOS.items():
  if a.only and a.only!=name:continue
  dst=ROOT/'external'/name
  if not dst.exists():
   dst.parent.mkdir(exist_ok=True);subprocess.run(['git','clone','--filter=blob:none',url,str(dst)],check=True)
   subprocess.run(['git','-C',str(dst),'checkout',sha],check=True)
  actual=subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip()
  if actual!=sha:raise RuntimeError(f'{dst}: expected {sha}, got {actual}; existing checkout left intact')
  if name=='verl':
   patch=str(ROOT/'patches/verl-sft.patch')
   done=subprocess.run(['git','-C',str(dst),'apply','--reverse','--check',patch],capture_output=True).returncode==0
   if not done:
    subprocess.run(['git','-C',str(dst),'apply','--check',patch],check=True);subprocess.run(['git','-C',str(dst),'apply',patch],check=True)
  if name=='AgentGym':
   for source,target in [('agentenv-babyai/agentenv_babyai','babyai_native_v1'),('agentenv-textcraft','textcraft_native_v1')]:
    out=ROOT/'external'/target
    if not out.exists():shutil.copytree(dst/source,out)
 print('Pinned external source setup complete. Install dependencies as documented.')
if __name__=='__main__':main()
