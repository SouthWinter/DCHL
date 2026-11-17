"""The execution certificate determines both supervision and feedback."""
from alfworld_bounded_hsl import BoundedHSLBuffer

class EvidenceBuffer(BoundedHSLBuffer):
    feedback_metric_key='execution_evidence'

def mixed_records(demonstrations,buffer,rng,**kwargs):
    from alfworld_hsl_batch import mixed_records as base_mixed_records
    rows,provenance=base_mixed_records(demonstrations,buffer,rng,**kwargs)
    counts={}
    for row in rows:
        draw=row.get('sampling_draw_id',-1)
        if draw<0:continue
        feedback=buffer.last_selected[draw]['goal']['feedback_steps']
        if row['step'] not in feedback:row['sampling_draw_id']=-1
        else:counts[draw]=counts.get(draw,0)+1
    for item in provenance:
        if item['source']=='hindsight':
            draw=item['sampling_draw_id'];assert counts.get(draw,0)>0
            item['feedback_steps']=buffer.last_selected[draw]['goal']['feedback_steps']
            item['feedback_actions']=counts[draw]
    return rows,provenance
