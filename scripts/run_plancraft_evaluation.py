"""Load one evaluator replica per available allocated GPU and clean up owned services."""
import argparse,json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=__import__('sys').executable
LOCAL_HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--run',required=True);p.add_argument('--gpus',default='0,3,4,5,6');p.add_argument('--split',default='test',choices=['val','test']);p.add_argument('--date-string',default=None);p.add_argument('--examples-file',default=None);a=p.parse_args()
    examples_file=Path(a.examples_file) if a.examples_file else ROOT/'data/plancraft_text_v1'/f'{a.split}.jsonl'
    expected_ids={json.loads(line)['id'] for line in examples_file.read_text().splitlines() if line.strip()}
    run=Path(a.run);logs=run/'logs';logs.mkdir(parents=True,exist_ok=True);processes=[];handles=[];endpoints=[];started=time.monotonic()
    try:
        for index,gpu in enumerate(a.gpus.split(',')):
            free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=24576,(gpu,free)
            uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            port=29910+index
            with socket.socket() as sock:
                # Match the HTTP server's reuse policy: a closed predecessor can
                # leave TIME_WAIT sockets without an active listener.
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                sock.bind(('127.0.0.1',port))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,VLLM_USE_V1='1',ALFWORLD_NVML_PHYSICAL_INDICES=gpu,LD_PRELOAD=str(ROOT/'cache/nvml_selected_devices.so'),OMP_NUM_THREADS='2',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
            log=(logs/f'server_gpu{gpu}.log').open('a');handles.append(log)
            cmd=[PY,'-m','vllm.entrypoints.openai.api_server','--model',a.model,'--served-model-name','plancraft-agent','--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--max-model-len','12288','--max-num-seqs','16','--gpu-memory-utilization',os.environ.get('PLANCRAFT_EVAL_GPU_MEMORY_UTILIZATION','0.4'),'--enforce-eager','--disable-log-requests']
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);processes.append(proc)
            endpoint=f'http://127.0.0.1:{port}';deadline=time.monotonic()+600
            while time.monotonic()<deadline:
                if proc.poll() is not None:raise RuntimeError(f'GPU{gpu} evaluator startup failed; inspect server log')
                try:
                    with LOCAL_HTTP.open(endpoint+'/health',timeout=2):break
                except Exception:time.sleep(2)
            else:raise TimeoutError(f'GPU{gpu} evaluator startup')
            endpoints.append(endpoint);print('ready',gpu,flush=True)
        env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts')+':'+str(ROOT/'external/plancraft'),NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost',OMP_NUM_THREADS='2')
        evaluate_command=[PY,str(ROOT/'scripts/evaluate_plancraft_text.py'),'--endpoints',','.join(endpoints),'--output',str(run/'outputs'),'--split',a.split,'--tokenizer',a.model]
        if a.date_string is not None:evaluate_command+=['--date-string',a.date_string]
        if a.examples_file is not None:evaluate_command+=['--examples-file',str(examples_file)]
        subprocess.run(evaluate_command,env=env,check=True)
        result=json.loads((run/'results.json').read_text());assert result['status']=='PASS' and result['episodes']==len(expected_ids)
        actual={json.loads(path.read_text())['id'] for path in (run/'outputs').glob('*.json')}
        assert actual==expected_ids, 'Evaluation task coverage differs from selected split'
        (run/'COMPLETE').touch()
    finally:
        for proc in processes:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        for proc in processes:
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for handle in handles:handle.close()
        (run/'runtime.json').write_text(json.dumps(dict(gpus=a.gpus,split=a.split,model=a.model,wall_seconds=time.monotonic()-started))+'\n')
