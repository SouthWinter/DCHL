import json
import pytest
from alfworld_soft_rehearsal import SoftRehearsalBuffer,family
from alfworld_odm import ODMController

def fixture():
    b=SoftRehearsalBuffer()
    for c,obj,w in [('a','easy',4.),('a','hard',2.),('b','new',1.)]:
        b.demonstrations.append(dict(rollout=dict(policy_version='v',gamefile=obj),
            goal=dict(task_type=c,object_type=obj,instruction=obj,prefix_length=2,hsl_weight=w),collection_round=0))
    return b

def feedback(b):
    a,h,_=b.demonstrations
    b.last_selected=[a,a,h]
    return b.observe_training(dict(odm_tokens_0=1,odm_nll_0=.2,odm_tokens_1=1,odm_nll_1=.2,odm_tokens_2=1,odm_nll_2=1.8))

def test_two_rounds_and_duplicate_draws_count_once():
    b=fixture();a=b.demonstrations[0]
    b.round_index=1;feedback(b)
    assert b.ease[family(a)]['observations']==1
    assert b.factor(a)==1
    b.round_index=2;feedback(b)
    assert b.factor(a)==pytest.approx(.6)
    assert b.factor(b.demonstrations[1])==1
    b.round_index=13
    assert b.factor(a)==pytest.approx(.8)

def test_actual_class_mass_and_resume():
    b=fixture();b.round_index=1;feedback(b);b.round_index=2;feedback(b)
    restored=SoftRehearsalBuffer.from_state_dict(json.loads(json.dumps(b.state_dict())))
    logits=ODMController().logits(b,3)
    a=b.sample(256,logits);assert a==restored.sample(256,logits)
    probs={x['goal']['object_type']:x['sample_probability'] for x in a}
    assert probs['easy']+probs['hard']==pytest.approx(.5)
    assert probs['new']==pytest.approx(.5)
    assert probs['easy']/probs['hard']==pytest.approx(1.2)
    assert b.last_sample_distribution==pytest.approx({'a':.5,'b':.5})

def test_hard_feedback_resets_suppression_and_zero_tokens_rejected():
    b=fixture();b.round_index=1;feedback(b);b.round_index=2;feedback(b)
    b.last_selected=list(b.demonstrations)[:2]
    b.observe_training(dict(odm_tokens_0=1,odm_nll_0=2,odm_tokens_1=1,odm_nll_1=.1))
    assert b.factor(b.demonstrations[0])==1
    with pytest.raises(ValueError):b.observe_training(dict(odm_tokens_0=0,odm_nll_0=0))
