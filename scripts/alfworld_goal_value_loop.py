"""One bounded demand estimate controls goal retention and repeated exposure."""
import math
import random
from collections import defaultdict,deque
from alfworld_hsl_buffer import HSLBuffer
from alfworld_soft_rehearsal import SoftRehearsalBuffer,family
from alfworld_odm import ODMController


class GoalValueBuffer(SoftRehearsalBuffer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.last_admission={}
        self.admission_rng=random.Random(kwargs.get('seed',42)+7919)

    def factor(self,entry):
        state=self.ease.get(family(entry))
        if state is None or state['observations']<2:return 1.
        age=max(0,self.round_index-state['last_round']-1)
        value=max(.5,min(2.,state['ema_ratio']))
        return 1+(value-1)*2**(-age/10)

    def weights(self,entries,retention=False):
        masses=defaultdict(float)
        for e in entries:masses[e['goal']['task_type']]+=e['goal']['hsl_weight']
        return [e['goal']['hsl_weight']*self.factor(e)/masses[e['goal']['task_type']]
            *(2**(-max(0,self.round_index-1-e['collection_round'])/2) if retention else 1.) for e in entries]

    def append_round(self,results,policy_version):
        # Reuse original validation and preserve every new candidate before selection.
        fresh=HSLBuffer(capacity=max(1,sum(len(r['goals']) for r in results)),allow_engineering=self.allow_engineering)
        fresh.append_round(results,policy_version)
        for e in fresh.demonstrations:e['collection_round']=self.round_index
        candidates=[e for e in self.demonstrations if self.round_index-e['collection_round']<3]+list(fresh.demonstrations)
        self.round_index+=1
        weights=self.weights(candidates,retention=True)
        # Weighted reservoir over candidates, followed by ordinary with-replacement
        # training draws. No within-batch exclusion of goals or source trajectories.
        admission_rng=self.admission_rng if getattr(self,'stable_data_schedule',False) else self.rng
        priorities=[math.log(max(admission_rng.random(),1e-300))/w for w in weights]
        selected=sorted(sorted(range(len(candidates)),key=lambda i:priorities[i],reverse=True)[:self.capacity])
        self.demonstrations=deque((candidates[i] for i in selected),maxlen=self.capacity)
        self.current_policy_version=policy_version
        self.last_admission=dict(candidates=len(candidates),retained=len(selected),
            new_candidates=len(fresh.demonstrations),new_retained=sum(candidates[i]['collection_round']==self.round_index-1 for i in selected),
            selected=[dict(policy_version=candidates[i]['rollout']['policy_version'],task=candidates[i]['rollout']['gamefile'],
                goal=candidates[i]['goal']['instruction'],prefix_length=candidates[i]['goal']['prefix_length'],
                family=family(candidates[i]),value=self.factor(candidates[i]),priority_weight=weights[i]) for i in selected])

    def state_dict(self):
        return dict(super().state_dict(),admission_rng=self.admission_rng.getstate())

    @classmethod
    def from_state_dict(cls,state):
        b=super().from_state_dict(state)
        if 'admission_rng' in state:
            def tuples(x):return tuple(map(tuples,x)) if isinstance(x,(list,tuple)) else x
            b.admission_rng.setstate(tuples(state['admission_rng']))
        return b

    def sample(self,count,goal_logits=None):
        entries=list(self.demonstrations);self.last_selected=[];self.last_sample_distribution={}
        if not entries:return []
        weights=self.weights(entries);denominator=sum(weights)
        for e,w in zip(entries,weights):
            c=e['goal']['task_type'];self.last_sample_distribution[c]=self.last_sample_distribution.get(c,0.)+w/denominator
        indices=self.rng.choices(range(len(entries)),weights=weights,k=count)
        self.last_selected=[entries[i] for i in indices]
        factors=[self.factor(e) for e in entries]
        self.last_rehearsal=dict(cache_entries=len(entries),mean_factor=sum(factors)/len(factors),
            min_factor=min(factors),max_factor=max(factors),adjusted_entries=sum(f!=1 for f in factors),
            admission=self.last_admission,draws=[dict(draw=d,family=family(entries[i]),factor=factors[i]) for d,i in enumerate(indices)])
        return [dict(rollout=entries[i]['rollout'],goal=entries[i]['goal'],sample_probability=weights[i]/denominator,
            cache_source_id=entries[i].get('cache_source_id'),policy_version=entries[i]['rollout']['policy_version'],
            collection_round=entries[i]['collection_round']) for i in indices]


class BufferValueController(ODMController):
    """Keep trainer per-draw NLL instrumentation; the buffer owns all allocation."""
    def logits(self,buffer,step):return {}

    def observe(self,step,provenance,probabilities,metrics):
        return dict(step=step,probabilities=probabilities,allocation='goal_value_buffer')
