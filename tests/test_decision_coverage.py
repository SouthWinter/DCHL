import json
import pytest
from alfworld_decision_coverage import DecisionCoverageBuffer
from alfworld_stable_evidence import source_key

def entry(support,source='episode-a'):
    return dict(rollout=dict(policy_version='online:step-1',gamefile=source,rollout_id=0),
       goal=dict(task_type='task',object_type='object',instruction='goal',feedback_steps=support,hsl_weight=1),collection_round=0)

def test_shared_credit_conserves_three_independent_decisions():
    b=DecisionCoverageBuffer();rows=[entry([0,1]),entry([0,1]),entry([0,2])]
    w=b.goal_weights(rows,[])
    assert w==pytest.approx([5/6,5/6,4/3])
    assert sum(w)==pytest.approx(3)
    assert [x/sum(w) for x in w]==pytest.approx([5/18,5/18,4/9])

def test_reserved_and_committed_exposure_agree_and_resume():
    b=DecisionCoverageBuffer();rows=[entry([0,1]),entry([0,1]),entry([0,2])]
    reserved=b.goal_weights(rows,[rows[2]])
    b.decision_exposure[source_key(rows[0])]={'0':1,'2':1}
    assert b.goal_weights(rows,[])==pytest.approx(reserved)
    restored=DecisionCoverageBuffer.from_state_dict(json.loads(json.dumps(b.state_dict())))
    assert restored.goal_weights(rows,[])==pytest.approx(reserved)
    assert reserved==pytest.approx([2/3]*3)

def test_exposure_from_another_source_does_not_reduce_credit():
    b=DecisionCoverageBuffer();rows=[entry([0,1])]
    assert b.goal_weights(rows,[entry([0,1],'episode-b')])==b.goal_weights(rows,[])
