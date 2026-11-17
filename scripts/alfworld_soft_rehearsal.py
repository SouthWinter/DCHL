"""Preserve ODM class mass and soften exposure of repeatedly easy goal families."""
import json
import math
from collections import defaultdict
from alfworld_hsl_buffer import HSLBuffer
from alfworld_hsl_credit import tilted_weights
from alfworld_odm import total
from alfworld_odm_coverage import demonstration_key


def family(entry):
    g=entry['goal']
    return json.dumps([g['task_type'],g['object_type']],ensure_ascii=False)


class SoftRehearsalBuffer(HSLBuffer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ease={}
        self.last_selected=[]
        self.last_rehearsal={}

    def factor(self,entry):
        state=self.ease.get(family(entry))
        if state is None or state['easy_streak']<2:return 1.
        age=max(0,self.round_index-state['last_round']-1)
        # Repeated low relative NLL softly reduces exposure; stale evidence fades.
        reduction=.5*(1-min(1.,state['ema_ratio']))*2**(-age/10)
        return 1-reduction

    def sample(self,count,goal_logits=None):
        entries=list(self.demonstrations)
        self.last_selected=[];self.last_sample_distribution={}
        if not entries:return []
        classes=[e['goal']['task_type'] for e in entries]
        original=tilted_weights([e['goal']['hsl_weight'] for e in entries],classes,goal_logits)
        factors=[self.factor(e) for e in entries]
        mass=defaultdict(float);adjusted_mass=defaultdict(float)
        for c,w,f in zip(classes,original,factors):mass[c]+=w;adjusted_mass[c]+=w*f
        weights=[w*f*mass[c]/adjusted_mass[c] for c,w,f in zip(classes,original,factors)]
        denominator=sum(weights)
        self.last_sample_distribution={c:v/sum(mass.values()) for c,v in mass.items()}
        indices=self.rng.choices(range(len(entries)),weights=weights,k=count)
        self.last_selected=[entries[i] for i in indices]
        self.last_rehearsal=dict(cache_entries=len(entries),suppressed_entries=sum(f<1 for f in factors),
            mean_factor=sum(factors)/len(factors),min_factor=min(factors),
            draws=[dict(draw=d,family=family(entries[i]),factor=factors[i]) for d,i in enumerate(indices)])
        return [dict(rollout=entries[i]['rollout'],goal=entries[i]['goal'],
            sample_probability=weights[i]/denominator,cache_source_id=entries[i].get('cache_source_id'),
            policy_version=entries[i]['rollout']['policy_version'],collection_round=entries[i]['collection_round']) for i in indices]

    def observe_training(self,metrics):
        # One observation per unique demonstration per update; repeated draws
        # never count as evidence from multiple training rounds.
        unique=defaultdict(list);lookup={}
        for draw,e in enumerate(self.last_selected):
            tokens=total(metrics[f'odm_tokens_{draw}'])
            if tokens<=0:raise ValueError('Missing rehearsal reward tokens')
            loss=total(metrics[f'odm_nll_{draw}'])/tokens
            if not math.isfinite(loss) or loss<0:raise ValueError('Invalid rehearsal loss')
            key=demonstration_key(e);lookup[key]=e;unique[key].append(loss)
        grouped=defaultdict(list);classes=defaultdict(list)
        for key,losses in unique.items():
            e=lookup[key];loss=sum(losses)/len(losses)
            grouped[family(e)].append(loss);classes[e['goal']['task_type']].append(loss)
        updates=[]
        for key,losses in grouped.items():
            cls=json.loads(key)[0]
            baseline=sum(classes[cls])/len(classes[cls])
            ratio=(sum(losses)/len(losses))/baseline if baseline>1e-12 else 1.
            previous=self.ease.get(key)
            ema=ratio if previous is None else .7*previous['ema_ratio']+.3*ratio
            streak=(previous['easy_streak'] if previous else 0)+1 if ratio<1 else 0
            self.ease[key]=dict(ema_ratio=ema,easy_streak=streak,last_round=self.round_index,
                observations=(previous['observations'] if previous else 0)+1)
            updates.append(dict(family=key,relative_nll=ratio,**self.ease[key]))
        return dict(**self.last_rehearsal,feedback=updates)

    def state_dict(self):
        return dict(super().state_dict(),rehearsal_ease=self.ease)

    @classmethod
    def from_state_dict(cls,state):
        buffer=super().from_state_dict(state)
        buffer.ease=state.get('rehearsal_ease',{})
        return buffer
