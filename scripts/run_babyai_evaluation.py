"""Two owned vLLM replicas, complete public90 evaluation, automatic cleanup."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
from babyai_adapter import ROOT, DATA, CONTEXT

PY = __import__('sys').executable
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--gpus',default='4,5')
    p.add_argument('--examples',default=str(DATA/'test.jsonl'))
    p.add_argument('--expected-episodes',type=int,default=90)
    p.add_argument('--scope',default='AgentGym public BabyAI90; native instruction order and termination')
    a=p.parse_args()
    run=Path(a.run)
    (run/'logs').mkdir(parents=True,exist_ok=True)
    children=[]; handles=[]; endpoints=[]
    started=time.monotonic()
    try:
        for i,gpu in enumerate(a.gpus.split(',')):
            free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            assert free>=16384,(gpu,free)
            uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            port=30610+i
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1',port))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,VLLM_USE_V1='1',ALFWORLD_NVML_PHYSICAL_INDICES=gpu,
                LD_PRELOAD=str(ROOT/'cache/nvml_selected_devices.so'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
            handle=(run/'logs'/f'server_gpu{gpu}.log').open('a');handles.append(handle)
            cmd=[PY,'-m','vllm.entrypoints.openai.api_server','--model',a.model,'--served-model-name','textcraft-agent',
                '--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--max-model-len',str(CONTEXT),
                '--max-num-seqs','48','--gpu-memory-utilization',os.environ.get('BABYAI_EVAL_GPU_MEMORY_UTILIZATION','0.3'),'--enforce-eager','--disable-log-requests',
                '--chat-template',str(DATA/'fixed_chat_template.jinja')]
            proc=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(proc)
            endpoint=f'http://127.0.0.1:{port}'
            deadline=time.monotonic()+600
            while time.monotonic()<deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'Evaluator GPU{gpu} failed')
                try:
                    with HTTP.open(endpoint+'/health',timeout=2):
                        break
                except (OSError,TimeoutError):
                    time.sleep(2)
            else:
                raise TimeoutError('Evaluator startup')
            endpoints.append(endpoint)
        subprocess.run([PY,str(ROOT/'scripts/evaluate_babyai.py'),'--model',a.model,
            '--examples',a.examples,'--endpoints',','.join(endpoints),'--run',str(run),
            '--temperature','0','--scope',a.scope],cwd=ROOT,check=True)
        result=json.loads((run/'results.json').read_text())
        assert result['episodes']==a.expected_episodes and result['infrastructure_errors']==0 and result['status']=='PASS'
        (run/'COMPLETE').touch()
    finally:
        for proc in children:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
        for proc in children:
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for handle in handles:handle.close()
        (run/'runtime.json').write_text(json.dumps(dict(model=a.model,gpus=a.gpus,wall_seconds=time.monotonic()-started)))


if __name__=='__main__':main()
