import json
import pytest
from alfworld_stable_evidence import StableEvidenceBuffer,maximal_certificates,source_key
from alfworld_soft_rehearsal import family

def entry(cls,proof,source='a',quality=10):
    return dict(rollout=dict(policy_version='online:step-0',gamefile=source,rollout_id=0),
        goal=dict(task_type=cls,object_type='apple',instruction=cls,
            prefix_length=max(proof)+1,feedback_steps=proof,hsl_weight=quality),collection_round=0)

def test_subsets_only_within_source_and_total_source_credit_is_bounded():
    a=entry('simple',[3]);b=entry('heat',[2,3]);c=entry('simple',[3],'other')
    assert maximal_certificates([a,b,c])==[b,c]
    b2=entry('clean',[1,3]);buf=StableEvidenceBuffer()
    weights=buf.base_weights([b,b2,c])
    assert sum(weights[:2])==pytest.approx(10)
    assert weights[2]==pytest.approx(10)
    assert sum(buf.base_weights([b,b2,b2]))==pytest.approx(10)

def test_sparse_feedback_shrinks_and_repeated_source_cannot_raise_confidence():
    b=StableEvidenceBuffer();e=entry('heat',[2,3]);key=family(e);b.round_index=5
    b.ease[key]=dict(observations=10,last_round=5,ema_ratio=2)
    b.feedback_sources[key]={source_key(e):5}
    assert b.factor(e)==1
    b.feedback_sources[key]['independent_source']=5
    assert b.factor(e)==pytest.approx(1.05)
    b.round_index=16
    assert b.factor(e)==1

def test_batch_coverage_marginals_and_resume():
    b=StableEvidenceBuffer(seed=4)
    b.demonstrations.extend([entry('simple',[3],quality=100),entry('heat',[2,3],'b',quality=1),entry('cool',[2,3],'c',quality=1)])
    restored=StableEvidenceBuffer.from_state_dict(json.loads(json.dumps(b.state_dict())))
    assert b.sample(32)==restored.sample(32)
    assert set(b.last_rehearsal['class_draws'])=={'simple','heat','cool'}
    assert b.last_rehearsal['class_draws']['simple']>20
    assert sum(b.last_sample_distribution.values())==pytest.approx(1)
    assert all(p>=1/32 for p in b.last_sample_distribution.values())

def test_cache_keeps_one_supported_condition_through_new_class_burst():
    b=StableEvidenceBuffer(capacity=3)
    b.demonstrations.append(entry('heat',[2,3],'old'))
    rows=[]
    for i in range(5):
        e=entry('simple',[3],str(i));r=dict(e['rollout'],action_relevance_ready=True,goals=[e['goal']]);rows.append(r)
    b.append_round(rows,'online:step-0')
    assert len(b.demonstrations)==3
    assert {e['goal']['task_type'] for e in b.demonstrations}=={'simple','heat'}
