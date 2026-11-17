"""Execution-labeled HSL adapter. Preserve original histories; mask irrelevant actions."""
import copy
from collections import Counter
from plancraft_adapter import TextSession,context_messages,replay_actions

def totals(inventory):
    result=Counter()
    for slot,item in inventory.items():
        if int(slot)!=0:result[item['type']]+=item['quantity']
    return result

def action_records(messages,tokenizer):
    records=[]
    for index,message in enumerate(messages):
        if message['role']!='assistant':continue
        prompt=context_messages(messages[:index],tokenizer,max_tokens=12000)
        row=copy.deepcopy(prompt+[message])
        for m in row:m['loss_on_this_turn']=False
        row[-1]['loss_on_this_turn']=True
        records.append(dict(messages=row,enable_thinking=False))
    return records

def relabel(example,trajectory,recipes,tokenizer,version,session=None,include_records=True):
    session=session or TextSession();initial=totals(example['slotted_inventory'])
    goals=[];seen=set();checks=0
    for step in trajectory:
        for target,quantity in totals(step['inventory_after']).items():
            # A new manufactured item, rather than relocation of initial stock.
            if target in seen or initial[target]>0 or quantity<=0 or target not in recipes:continue
            seen.add(target);length=step['step'];changed=copy.deepcopy(example);changed['target']=target
            actions=[r['action'] for r in trajectory[:length]]
            replay=replay_actions(session,changed,recipes[target],actions)
            if not replay['won'] or len(replay['trajectory'])!=length:raise ValueError('Goal onset differs on replay')
            # Greedy deletion determines relevance; learning retains ORIGINAL
            # contexts and prefix length, as required by the masked HSL loss.
            kept=list(range(length))
            for index in reversed(range(length)):
                candidate=[i for i in kept if i!=index]
                trial=replay_actions(session,changed,recipes[target],[actions[i] for i in candidate]);checks+=1
                if trial['won']:kept=candidate
            mask=[i in kept for i in range(length)];n=sum(mask)
            if not n:raise ValueError('New goal has empty relevance mask')
            records=action_records(replay['messages'][:-1],tokenizer) if include_records else []
            if include_records:assert len(records)==length
            goals.append(dict(instruction=target,task_type='craft',prefix_length=length,
                relevant_actions=n,action_loss_mask=mask,hsl_weight=(n/length)**.8*n,
                records=records,relevance_checks=length))
    return dict(gamefile=example['id'],policy_version=version,action_relevance_ready=True,
        goals=goals,relevance_checks=checks,label_source='Official inventory and execution deletion; public recipe')

def training_records(rollout,goal,mixture=1.,sampled_trajectories=1):
    rows=copy.deepcopy(goal['records'])
    for row,supervised in zip(rows,goal['action_loss_mask']):
        row.update(action_supervised=supervised,trajectory_action_coefficient=mixture/sampled_trajectories/goal['prefix_length'])
        row['messages'][-1]['loss_on_this_turn']=supervised
    return rows
