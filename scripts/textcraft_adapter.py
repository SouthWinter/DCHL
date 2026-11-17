"""Direct native TextCraft execution with explicit, frozen semantic tasks.

Load native modules without the package initializer, which starts an HTTP app
and constructs an unrelated global environment. Native transition code stays
unchanged. Task instructions carry their published recipe list and target.
"""
import copy
import os
import re
import sys
import types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('TEXTCRAFT_DATA_DIR',str(ROOT/'data/textcraft_v1')))
NATIVE=ROOT/'external/textcraft_native_v1/agentenv_textcraft'
sys.path.insert(0,str(ROOT/'external/textcraft_runtime'))
if '_textcraft_native' not in sys.modules:
    package=types.ModuleType('_textcraft_native');package.__path__=[str(NATIVE)]
    sys.modules[package.__name__]=package
from _textcraft_native.environment import TextCraftEnv
from _textcraft_native.crafting_tree import CraftingTree

MAX_STEPS=20
CONTEXT=8192
SYSTEM=('You control an agent in TextCraft. Complete the requested crafting task. '
        'Return exactly one command per turn, without explanation. Commands: '
        'get COUNT ITEM; craft COUNT ITEM using COUNT INGREDIENT, COUNT INGREDIENT; inventory. '
        'Use the provided crafting recipes. For a generic ingredient such as planks, '
        'a matching specific type such as oak planks is allowed. Specify quantities.')
_TREE=None

def tree():
    global _TREE
    if _TREE is None:_TREE=CraftingTree(str(NATIVE))
    return _TREE

def parse_action(text):
    matches=re.findall(r'Action:\s*(.*?)(?=\n|$)',text,re.DOTALL)
    action=matches[-1] if matches else text.strip().splitlines()[0] if text.strip() else ''
    # Match the official AgentGym client normalization.
    return ' '.join(re.sub(r'[^A-Za-z0-9, ]+','',action).split()).strip()

def from_demo(row):
    prompt=next(m['value'] for m in row['conversations'] if 'Crafting commands:' in m['value'])
    commands=prompt.split('Crafting commands:',1)[1].split('Goal:',1)[0].strip()
    name=re.search(r'Goal: craft (.*?)\.',prompt).group(1)
    goal='minecraft:'+name.replace(' ','_')
    return dict(id=row['item_id'],commands=commands,goal=goal,depth=tree().get_min_depth(goal))

def context_messages(messages,tokenizer,max_tokens=CONTEXT-192):
    rows=copy.deepcopy(messages)
    while len(tokenizer.apply_chat_template(rows,tokenize=True,add_generation_prompt=True))>max_tokens:
        if len(rows)<=4:raise ValueError('Initial TextCraft prompt exceeds context')
        del rows[2:4]
    return rows

def action_records(messages,tokenizer):
    result=[]
    for i,m in enumerate(messages):
        if m['role']!='assistant':continue
        rows=context_messages(messages[:i],tokenizer)+[copy.deepcopy(m)]
        for x in rows:x['loss_on_this_turn']=False
        rows[-1]['loss_on_this_turn']=True
        result.append(dict(messages=rows,enable_thinking=False))
    return result

class TextSession:
    def __init__(self):self.env=TextCraftEnv(tree(),'',None)
    def reset(self,example,goal=None):
        self.example=example;self.goal=goal or example['goal']
        observation,_=self.env.reset(commands=example['commands'],goal=self.goal)
        self.description='Goal: craft '+self.goal.replace('minecraft:','').replace('_',' ')+'.'
        self.messages=[dict(role='system',content=SYSTEM),dict(role='user',content=observation)]
        self.trajectory=[];self.won=False;self.done=False;self.score=0.
        return observation
    def step(self,response):
        action=parse_action(response)
        observation,reward,terminated,truncated,_=self.env.step(action)
        self.done=terminated or truncated;self.won=bool(reward==1);self.score=100.*reward
        valid=observation.startswith(('Got ','Crafted ','Inventory:'))
        self.trajectory.append(dict(step=len(self.trajectory)+1,action=action,response=response,
            observation=observation,valid=valid,won=self.won,score=self.score,inventory=dict(self.env.inventory)))
        self.messages.extend([dict(role='assistant',content=action),dict(role='user',content=observation)])
        return observation,self.won
    def replay(self,example,actions,goal=None):
        self.reset(example,goal)
        for action in actions:
            self.step(action)
            if self.done:break
        return dict(won=self.won,score=self.score,description=self.description,
            trajectory=copy.deepcopy(self.trajectory),messages=copy.deepcopy(self.messages))
    def close(self):self.env.close()

def native_support(session,example,actions,goal):
    full=session.replay(example,actions,goal)
    if not full['won']:return None
    keep=list(range(len(full['trajectory'])))
    for index in reversed(keep.copy()):
        trial=[i for i in keep if i!=index]
        if session.replay(example,[actions[i] for i in trial],goal)['won']:keep=trial
    assert keep and session.replay(example,[actions[i] for i in keep],goal)['won']
    return dict(full=full,feedback_steps=keep)

def training_records(rollout,goal,mixture=1.,sampled_trajectories=1):
    rows=copy.deepcopy(goal['records'])
    assert len(rows)==len(goal['action_loss_mask'])==goal['prefix_length']
    for i,(row,active) in enumerate(zip(rows,goal['action_loss_mask'])):
        row.update(action_supervised=active,trajectory_action_coefficient=mixture/sampled_trajectories/goal['prefix_length'],
            policy_version=rollout['policy_version'],source_episode_id=rollout['gamefile'],step=i)
        row['messages'][-1]['loss_on_this_turn']=active
    return rows
