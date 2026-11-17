import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from alfworld_action_contract import canonical_messages,mask_repeated_navigation

def test_canonical_targets_preserve_actions_and_remove_old_reasoning():
    source=[dict(role='user',content='old'),dict(role='assistant',content='OK'),dict(role='user',content='goal'),dict(role='assistant',content='Thought: old goal\nAction: take mug 1 from shelf 1',loss_on_this_turn=True)]
    original=copy.deepcopy(source)
    result=canonical_messages(source)
    assert source==original
    assert result[-1]['content']=='Action: take mug 1 from shelf 1'
    assert result[-1]['loss_on_this_turn']
    assert 'Thought:' not in str(result)

def test_refinement_preserves_first_observation_and_changed_observation():
    def step(dest,origin,observation=None):
        return dict(action='go to '+dest,result_observation=observation or dest,added_facts=[['at',dest]],removed_facts=[['at',origin]])
    steps=[step('a','start'),step('b','a'),step('a','b'),step('b','a'),step('a','b')]
    goal=dict(prefix_length=5,action_loss_mask=[True]*5,relevance_reasons=['support']*5)
    revised=mask_repeated_navigation(steps,goal)
    assert revised['action_loss_mask']==[True,True,False,False,True]
    assert goal['action_loss_mask']==[True]*5
    assert revised['hsl_weight']==(3/5)**.8*3
    steps[3]['result_observation']='new object'
    assert mask_repeated_navigation(steps,goal)['action_loss_mask']==[True]*5

def test_hindsight_contract_overrides_legacy_thought_span(monkeypatch):
    monkeypatch.setenv("ALFWORLD_ACTION_CONTRACT", "1")
    from alfworld_dialogue_hsl import hindsight_records
    rollout=dict(response_contract='action_only_v1',dialogue_supervision='preserve_response_action_span',
        policy_version='test',gamefile='fixture',steps=[dict(observation='Your task is to: heat a mug.',
        result_observation='You pick up mug 1.',action='take mug 1 from shelf 1',
        response='Thought: I should heat the mug.\nAction: take mug 1 from shelf 1')])
    goal=dict(prefix_length=1,instruction='put a mug in/on the desk.',task_type='pick_and_place_simple',action_loss_mask=[True])
    row=hindsight_records(rollout,goal,.3,32)[0]
    assert row['messages'][-1]['content']=='Action: take mug 1 from shelf 1'
    assert 'Thought:' not in str(row['messages'])
    assert 'heat a mug' not in str(row['messages'])
    assert row['trajectory_action_coefficient']==.3/32
