"""FIFO HSL quality sampling with a bounded relative-NLL multiplier."""
from collections import defaultdict
from alfworld_soft_rehearsal import SoftRehearsalBuffer,family


class BoundedHSLBuffer(SoftRehearsalBuffer):
    feedback_metric_key='bounded_hsl'

    def factor(self,entry):
        state=self.ease.get(family(entry))
        if state is None or state['observations']<2:return 1.
        age=max(0,self.round_index-state['last_round']-1)
        # Same EMA relative-NLL feedback as the prior loop; narrower declared
        # multiplier and unchanged FIFO. Old evidence fades toward neutrality.
        value=max(.8,min(1.25,state['ema_ratio']))
        return 1+(value-1)*2**(-age/10)

    def sample(self,count,goal_logits=None):
        if goal_logits:raise ValueError('Bounded HSL preserves the base class mass; external class logits are unsupported')
        entries=list(self.demonstrations)
        self.last_selected=[];self.last_sample_distribution={};self.last_rehearsal={}
        if not entries:return []
        base=[e['goal']['hsl_weight'] for e in entries]
        factors=[self.factor(e) for e in entries]
        weights=[w*f for w,f in zip(base,factors)]
        denominator=sum(weights);base_total=sum(base);base_distribution=defaultdict(float)
        for e,w,b in zip(entries,weights,base):
            c=e['goal']['task_type']
            self.last_sample_distribution[c]=self.last_sample_distribution.get(c,0.)+w/denominator
            base_distribution[c]+=b/base_total
        indices=self.rng.choices(range(len(entries)),weights=weights,k=count)
        self.last_selected=[entries[i] for i in indices]
        self.last_rehearsal=dict(cache_entries=len(entries),mean_factor=sum(factors)/len(factors),
            min_factor=min(factors),max_factor=max(factors),adjusted_entries=sum(f!=1 for f in factors),
            base_class_probabilities=dict(base_distribution),class_probabilities=dict(self.last_sample_distribution),
            class_total_variation=.5*sum(abs(base_distribution[c]-self.last_sample_distribution[c]) for c in base_distribution),
            draws=[dict(draw=d,family=family(entries[i]),factor=factors[i]) for d,i in enumerate(indices)])
        return [dict(rollout=entries[i]['rollout'],goal=entries[i]['goal'],sample_probability=weights[i]/denominator,
            cache_source_id=entries[i].get('cache_source_id'),policy_version=entries[i]['rollout']['policy_version'],
            collection_round=entries[i]['collection_round']) for i in indices]
