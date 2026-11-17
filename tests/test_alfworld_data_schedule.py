import json
import random
import pytest
from alfworld_data_schedule import round_tasks,row_rng,completed_rows,check_resume_schedule
from alfworld_goal_value_loop import GoalValueBuffer


def test_tasks_independent_of_row_count_and_resume_position():
    streams=[random.Random(42),random.Random(42)]
    for step in range(1,76):
        tasks=[round_tasks(list(range(800)),18,42,step,r,True) for r in streams]
        assert tasks[0]==tasks[1]
        for r,n in zip(streams,(715,775)):
            row_rng(42,step,r,True).shuffle(list(range(n)))
            r.shuffle(list(range(n)))
    assert round_tasks(list(range(800)),18,42,60,random.Random(999),True)==round_tasks(list(range(800)),18,42,60,streams[0],True)


def test_stable_completion_order_and_legacy_semantics():
    rs=[dict(rollout_id=i,status='complete') for i in (2,0,1)]
    assert [r['rollout_id'] for r in completed_rows(rs,3,True)]==[0,1,2]
    assert completed_rows(rs,3,False)==rs
    assert completed_rows(list(reversed(rs)),3,True)==completed_rows(rs,3,True)
    with pytest.raises(ValueError):completed_rows(rs[:2],3,True)
    with pytest.raises(ValueError):check_resume_schedule({},True)
    check_resume_schedule({},False);check_resume_schedule({'stable_data_schedule':True},True)


def test_stable_admission_rng_independent_and_restored():
    b=GoalValueBuffer(capacity=3,seed=42);b.stable_data_schedule=True
    def rs(v):
        return [dict(policy_version=v,gamefile='task',action_relevance_ready=True,goals=[
            dict(task_type='a',object_type='apple',instruction=f'g{i}',prefix_length=i+1,hsl_weight=i+1.) for i in range(10)])]
    state=b.rng.getstate();b.append_round(rs('v1'),'v1');assert b.rng.getstate()==state
    b.sample(32)
    c=GoalValueBuffer.from_state_dict(json.loads(json.dumps(b.state_dict())));c.stable_data_schedule=True
    b.append_round(rs('v2'),'v2');c.append_round(rs('v2'),'v2')
    assert b.last_admission==c.last_admission
    assert b.sample(32)==c.sample(32)
