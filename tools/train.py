"""Launch the released online recipes from an explicit common SFT checkpoint."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def build(a):
    c=json.loads((ROOT/'configs'/f'{a.dataset}_{a.model}.json').read_text())
    model=Path(a.sft).resolve();run=Path(a.run).resolve();data=ROOT/'data'/c['data_directory']
    env=dict(os.environ,DCHL_ROOT=str(ROOT),DCHL_PYTHON=sys.executable,
        ALFWORLD_INITIAL_MODEL_PATH=str(model),ALFWORLD_SFT_MODULE=c['modules'][a.method],
        ALFWORLD_SFT_LR=str(c['learning_rate']),ALFWORLD_MODEL_REMOVE_PADDING='false',
        ALFWORLD_TRAIN_MICRO_BATCH_SIZE=str(c['microbatch']),ALFWORLD_SAVE_FREQ=str(c['save_frequency']),
        ALFWORLD_MIN_FREE_GIB=str(a.min_free_gib),ALFWORLD_TRAIN_SHARED_GPU='0',
        VES_CHECKPOINT_OVERWRITE_PREVIOUS='1',ALFWORLD_DATA_SHUFFLE='false' if a.dataset=='alfworld' else 'true')
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'scripts'),os.environ.get('DCHL_VERL_ROOT',str(ROOT/'external/verl')),str(ROOT/'external/plancraft'),env.get('PYTHONPATH','')])
    if a.dataset=='alfworld':
        env.pop('ALFWORLD_HINDSIGHT_BACKEND',None)
        if a.method!='hsl':env['ALFWORLD_HINDSIGHT_BACKEND']='observable_evidence_v1'
        if a.model=='qwen':env['ALFWORLD_ISOLATE_SAMPLES']='1'
        env.update(ALFWORLD_ACTION_CONTRACT='1',ALFWORLD_DIALOGUE_HSL='1',ALFWORLD_ETO_MAP_PUT='1',ALFWORLD_DIALOGUE_SUPERVISION='action_only',ALFWORLD_HSL_SAMPLER='base' if a.method=='hsl' else 'odm')
    else:
        template=Path(a.chat_template) if a.chat_template else data/'fixed_chat_template.jinja'
        env.update(ALFWORLD_FIXED_CHAT_TEMPLATE=str(template),ALFWORLD_ISOLATE_SAMPLES='1')
    sampler='base' if a.method=='hsl' else 'odm'
    args=['bash',str(ROOT/'scripts/run_alfworld_sft_stage_4gpu.sh'),str(data/'hsl_demonstrations.parquet'),str(c['updates']),str(run/'train'),str(run/'training_state/checkpoints'),'auto' if a.resume else 'disable','',a.gpus,
      f'data.train_batch_size={c["physical_batch_size"]}',f'data.max_length={c["max_length"]}',f'data.max_token_len_per_gpu={c["max_length"]}',f'trainer.seed={c["seed"]}',
      'checkpoint.save_contents=[optimizer,extra,hf_model]','checkpoint.load_contents=[optimizer,extra]',
      f'+hsl.output_dir={run}/online',f'+hsl.endpoint=http://127.0.0.1:{a.port}','+hsl.start_service=true',
      '+hsl.service_memory_gib=12' if a.dataset=='babyai' else '+hsl.service_memory_gib=14',
      f'+hsl.max_model_len={c["max_length"]}',f'+hsl.sampler={sampler}','+hsl.stable_data_schedule=true',
      f'+hsl.task_pool={ROOT}/data/{c["task_pool"]}','+hsl.rollouts_per_step=18',f'+hsl.workers={c["workers"]}',
      '+hsl.buffer_capacity=100','+hsl.batch_trajectories=32','+hsl.hindsight_fraction=0.3']
    if a.dataset=='alfworld':args += [f'+hsl.environment_python={a.environment_python}',f'+hsl.data_root={Path(os.environ["ALFWORLD_DATA"]).parent}','+hsl.relevance_rule=backward']
    if a.dataset=='babyai':args += ['+hsl.service_max_num_seqs=24']
    return c,env,args

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['alfworld','plancraft','babyai'],required=True)
    p.add_argument('--model',choices=['llama','qwen'],required=True)
    p.add_argument('--method',choices=['hsl','dchl','dynamixsft','data_agent'],required=True)
    p.add_argument('--sft',required=True);p.add_argument('--run',required=True);p.add_argument('--gpus',required=True)
    p.add_argument('--chat-template');p.add_argument('--environment-python',default=sys.executable)
    p.add_argument('--port',type=int,default=30890);p.add_argument('--min-free-gib',type=int,default=60)
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();c,env,args=build(a)
    print(json.dumps(dict(config=c,command=args,environment={k:v for k,v in env.items() if k.startswith(('ALFWORLD_','VES_','DCHL_'))}),indent=2))
    if a.dry_run:return
    model=Path(a.sft);run=Path(a.run)
    if not (model/'config.json').is_file():p.error('SFT checkpoint is missing config.json')
    if not a.resume and (run/'training_state/checkpoints/latest_checkpointed_iteration.txt').exists():p.error('Run has a checkpoint; use --resume or a new directory')
    if a.resume:
        tracker=run/'training_state/checkpoints/latest_checkpointed_iteration.txt'
        if not tracker.is_file():p.error('--resume requires an existing checkpoint tracker')
        step=int(tracker.read_text());state=run/f'training_state/checkpoints/global_step_{step}/online_state.json'
        if not state.is_file() or json.loads(state.read_text())['step']!=step:p.error('Incomplete online checkpoint')
        metrics=run/'online/metrics.jsonl'
        if metrics.exists() and any(json.loads(l)['step']>step for l in metrics.read_text().splitlines() if l.strip()):p.error('Uncommitted metrics after checkpoint: archive that tail before resuming (docs/reproduction.md)')
    run.mkdir(parents=True,exist_ok=True)
    if a.dataset=='alfworld' and a.method!='hsl':
        import pandas as pd
        rows=pd.read_parquet(ROOT/'data/alfworld_action_contract_v1/hsl_demonstrations.parquet');bank=[]
        for key,g in rows.groupby('source_episode_id'):
            messages=g.sort_values('step_index').iloc[-1]['messages'];goal=messages[2]['content'].split('Your task is to: ')[-1].split('\n')[0]
            actions=[m['content'] for m in messages if m['role']=='assistant' and m['content'].startswith('Action:')]
            bank.append(dict(id=str(key),goal=goal,context='\n'.join(m['content'] for m in messages if m['role']=='user'),text='Task: '+goal+'\n'+'\n'.join(actions)))
        (run/'recovery_reference_bank.json').write_text(json.dumps(bank))
    (run/'release_config.json').write_text(json.dumps(c,indent=2))
    subprocess.run(args,env=env,cwd=ROOT,check=True)
if __name__=='__main__':main()
