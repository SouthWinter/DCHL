"""Full-dialogue Thought/Action evaluation over the existing task panels."""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'scripts/evaluate_alfworld_openai_panel.py'
spec=importlib.util.spec_from_file_location('eto_panel_base',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
module=base.module
prefix=json.loads((ROOT/'data/alfworld_eto_sft800_v1/instruction.json').read_text())

def messages(self):
    if not hasattr(self,'dialogue'):
        if os.environ.get('ALFWORLD_ACTION_CONTRACT')=='1':
            from alfworld_action_contract import prefix as action_prefix
            self.dialogue=action_prefix()
        else:
            self.dialogue=[dict(role={'human':'user','gpt':'assistant'}[m['from']],content=m['value']) for m in prefix]
        self.dialogue.append(dict(role='user',content=self.current_observation))
    return self.dialogue

def parse(response):
    match=re.search(r'(?:^|\n)Action:\s*([^\n]+)',response)
    return match.group(1).strip() if match else ''

def advance(self,response):
    messages(self)
    action=parse(response)
    raw_action=action
    if os.environ.get('ALFWORLD_ETO_MAP_PUT')=='1':
        match=re.fullmatch(r'put (.+) in/on (.+)',action)
        if match:
            mapped=f'move {match[1]} to {match[2]}'
            if mapped in self.admissible_actions:action=mapped
    valid=action.lower() in {a.lower() for a in self.admissible_actions}
    self.invalid_action_count+=int(not valid)
    # Send generated actions directly; invalid commands consume an environment step.
    state=self.episode.step(action or 'invalid action')
    self.total_reward+=state.reward;self.done=state.done;self.won=state.won;self.step_id+=1
    self.trajectory.append(dict(step=self.step_id,action=action,raw_action=raw_action,model_response=response,valid_action=valid,invalid_reason=None if valid else 'invalid_generated_action',reward=state.reward,done=self.done,won=self.won))
    self.finished=self.done or self.step_id>=self.task['max_steps']
    history_response=response
    if os.environ.get('ALFWORLD_ACTION_CONTRACT')=='1':
        from alfworld_dialogue_hsl import public_action
        history_response='Action: '+public_action(action or 'invalid action')
    self.dialogue.extend([dict(role='assistant',content=history_response),dict(role='user',content='Observation: '+state.observation)])
    self.current_observation=state.observation;self.admissible_actions=state.admissible_actions

class DialogueModel(base.LocalOpenAIChatModel):
    def generate(self,messages):
        # Keep the instruction and initial observation; evict oldest complete turns
        # only when the actual chat token count exceeds the8192 server budget.
        kept=list(messages)
        while True:
            payload=dict(model=self.model,messages=kept,add_generation_prompt=True)
            req=base.urllib.request.Request(self.url.replace('/v1/chat/completions','/tokenize'),data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with base.urllib.request.urlopen(req,timeout=self.timeout) as response:count=json.load(response)['count']
            if count<=7680:break
            if len(kept)<=5:raise ValueError('Initial and latest observation exceed context budget')
            del kept[3:5]
        self.guided_choice=False
        return super().generate(kept)

module.ActiveAlfWorldEpisode.messages=messages
module.ActiveAlfWorldEpisode.advance=advance
base.LocalOpenAIChatModel=DialogueModel

if __name__=='__main__':
    sys.argv=[x for x in sys.argv if x!='--guided-choice']
    if '--max-tokens' not in sys.argv:sys.argv.extend(['--max-tokens','512'])
    base.main()
