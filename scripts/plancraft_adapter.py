"""Text-only interface using the official PlanCraft transitions and action parser."""
import copy,json
from pathlib import Path
from plancraft.config import PlancraftExample
from plancraft.environment.env import PlancraftEnvironment,target_and_inventory_to_text_obs
from plancraft.environment.actions import MoveActionHandler,SmeltActionHandler
from plancraft.environment.prompts import get_system_prompt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/plancraft_text_v1'
HANDLERS=[MoveActionHandler(),SmeltActionHandler()]

def completed(inventory,target):
    return any(slot!=0 and item['type']==target and item['quantity']>0 for slot,item in inventory.items())

class TextSession:
    def __init__(self):self.env=PlancraftEnvironment(resolution='low')
    def reset(self,example,recipe):
        self.example=example if isinstance(example,PlancraftExample) else PlancraftExample(**example)
        self.env.reset(copy.deepcopy(self.example.slotted_inventory));self.steps=0
        observation=target_and_inventory_to_text_obs(self.example.target,self.env.state)
        self.messages=[get_system_prompt(),dict(role='user',content=observation+'\n\n'+recipe)]
        self.trajectory=[]
        return observation
    def step(self,response):
        before=copy.deepcopy(self.env.state);action=None;error=None
        for handler in HANDLERS:
            parsed=handler.match(response)
            if parsed is not None:
                if isinstance(parsed,str):error=parsed
                else:action=parsed
                break
        if action is None:
            error=error or 'Format Error: return one move or smelt action in the documented format.'
        else:self.env.step(action)
        after=copy.deepcopy(self.env.state);self.steps+=1
        observation=target_and_inventory_to_text_obs(self.example.target,after)
        if error:observation=error+'\n'+observation
        canonical=str(action) if action is not None else response
        won=completed(after,self.example.target)
        self.trajectory.append(dict(step=self.steps,action=canonical,response=response,inventory_before=before,inventory_after=after,observation=observation,valid=action is not None,changed=before!=after,won=won))
        self.messages.extend([dict(role='assistant',content=canonical),dict(role='user',content=observation)])
        return observation,won

def context_messages(messages,tokenizer,max_tokens=12160):
    selected=copy.deepcopy(messages)
    while len(tokenizer.apply_chat_template(selected,tokenize=True,add_generation_prompt=True))>max_tokens:
        if len(selected)<=4:raise ValueError('Initial PlanCraft instruction exceeds context budget')
        del selected[2:4]
    return selected

def replay_actions(session,example,recipe,actions):
    session.reset(example,recipe);won=False
    for action in actions:
        _,won=session.step(action)
        if won:break
    return dict(won=won,trajectory=copy.deepcopy(session.trajectory),messages=copy.deepcopy(session.messages))

def compress_success(session,example,recipe,actions,cut,max_checks=24):
    """Delete suffix actions only after re-execution confirms the original goal."""
    result=replay_actions(session,example,recipe,actions)
    if not result['won']:raise ValueError('Compression requires a successful trajectory')
    assert 0<=cut<len(result['trajectory'])
    actions=[r['action'] for r in result['trajectory']];original=len(actions);prefix=result['trajectory'][:cut];checks=0
    for index in reversed(range(cut,len(actions))):
        if checks>=max_checks:break
        candidate=actions[:index]+actions[index+1:]
        trial=replay_actions(session,example,recipe,candidate);checks+=1
        assert trial['trajectory'][:cut]==prefix
        if trial['won']:result=trial;actions=[r['action'] for r in trial['trajectory']]
    result.update(actions=actions,cut=cut,checks=checks,removed=original-len(actions))
    return result
