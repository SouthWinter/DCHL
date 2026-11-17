"""Bounded backward deletion with original-goal execution verification."""
import copy,time

def compress_result(result,data_root,max_checks=12):
    if result.get('status')!='complete' or not result.get('won'):return result
    import os,textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
    os.environ['ALFWORLD_DATA']=data_root
    started=time.monotonic();out=copy.deepcopy(result);cut=result['cut'];env=None;executed=0;checks=0
    actions=[s['action'] for s in result['trajectory'][cut:]]
    def trial(suffix):
        nonlocal executed
        state=env.reset();trajectory=[]
        for original in result['trajectory'][:cut]:
            if str(state.feedback)!=original['observation']:raise ValueError('Compression prefix mismatch')
            state,_,done=env.step(original['action']);executed+=1
            if str(state.feedback)!=original['result_observation']:raise ValueError('Compression transition mismatch')
            trajectory.append(copy.deepcopy(original))
            if done or state.won:raise ValueError('Unexpected terminal compression prefix')
        for action in suffix:
            observation=str(state.feedback);state,_,done=env.step(action);executed+=1
            trajectory.append(dict(step=len(trajectory),action=action,observation=observation,
                result_observation=str(state.feedback),won=bool(state.won),done=bool(done)))
            if done or state.won:break
        return bool(state.won),trajectory
    try:
        env=textworld.start(result['gamefile'],request_infos=textworld.EnvInfos(won=True,admissible_commands=True),wrappers=[AlfredDemangler(shuffle=False)])
        original_count=len(actions)
        for i in reversed(range(original_count)):
            if checks>=max_checks:break
            proposed=actions[:i]+actions[i+1:];won,trajectory=trial(proposed);checks+=1
            if won:
                out['trajectory']=trajectory;actions=[s['action'] for s in trajectory[cut:]]
        out.update(compression_removed=original_count-len(actions),compression_checks=checks,
            compression_steps=executed,compression_seconds=time.monotonic()-started)
    finally:
        if env is not None:env.close()
    return out
