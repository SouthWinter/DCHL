"""Certificate-level replay with source credit and supported-condition coverage."""
import json
from collections import Counter,defaultdict,deque
from alfworld_evidence_sampling import EvidenceBuffer
from alfworld_hsl_buffer import HSLBuffer
from alfworld_soft_rehearsal import family

def source_key(entry):
    r=entry['rollout']
    return json.dumps([r['policy_version'],r['gamefile'],r.get('rollout_id')])

def maximal_certificates(entries):
    groups=defaultdict(list)
    for e in entries:groups[source_key(e)].append(e)
    result=[]
    for group in groups.values():
        proofs=[set(e['goal']['feedback_steps']) for e in group]
        # Strict inclusion preserves distinct conditions supported by different
        # actions; identical event sets are not silently merged across goals.
        result.extend(e for i,e in enumerate(group) if not any(proofs[i]<p for p in proofs))
    return result

class StableEvidenceBuffer(EvidenceBuffer):
    feedback_metric_key='stable_execution_evidence'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.feedback_sources={}
        self.last_retention={}

    def append_round(self,results,policy_version):
        fresh=HSLBuffer(capacity=max(1,sum(len(r['goals']) for r in results)),allow_engineering=self.allow_engineering)
        fresh.append_round(results,policy_version)
        for e in fresh.demonstrations:e['collection_round']=self.round_index
        all_entries=list(self.demonstrations)+list(fresh.demonstrations)
        candidates=maximal_certificates(all_entries)
        latest={e['goal']['task_type']:i for i,e in enumerate(candidates)}
        if len(latest)>self.capacity:raise ValueError('Capacity cannot retain supported conditions')
        keep=set(latest.values())
        for i in reversed(range(len(candidates))):
            if len(keep)>=self.capacity:break
            keep.add(i)
        self.demonstrations=deque((candidates[i] for i in sorted(keep)),maxlen=self.capacity)
        self.round_index+=1;self.current_policy_version=policy_version
        self.last_retention=dict(input_entries=len(all_entries),maximal_entries=len(candidates),
            retained=len(keep),supported_classes=len(latest),subsumed=len(all_entries)-len(candidates))

    def factor(self,entry):
        sources=self.feedback_sources.get(family(entry),{})
        n=sum(self.round_index-r<=10 for r in sources.values())
        if n<2:return 1.
        return 1+(super().factor(entry)-1)*n/(n+8.)

    def base_weights(self,entries):
        groups=defaultdict(list)
        for i,e in enumerate(entries):groups[source_key(e)].append(i)
        weights=[0.]*len(entries)
        for indices in groups.values():
            frequency=Counter(j for i in indices for j in entries[i]['goal']['feedback_steps'])
            quality=[entries[i]['goal']['hsl_weight'] for i in indices]
            provisional=[]
            for i,q in zip(indices,quality):
                proof=entries[i]['goal']['feedback_steps']
                provisional.append(q*sum(1/frequency[j] for j in proof)/len(proof))
            scale=max(quality)/sum(provisional)
            for i,w in zip(indices,provisional):weights[i]=w*scale
        return weights

    def sample(self,count,goal_logits=None):
        if goal_logits:raise ValueError('External class logits unsupported')
        entries=list(self.demonstrations)
        self.last_selected=[];self.last_sample_distribution={};self.last_rehearsal={}
        if not entries:return []
        classes=defaultdict(list)
        for i,e in enumerate(entries):classes[e['goal']['task_type']].append(i)
        if count<len(classes):raise ValueError('Batch smaller than supported conditions')
        base=self.base_weights(entries);factors=[self.factor(e) for e in entries]
        weights=[b*f for b,f in zip(base,factors)];mass={c:sum(weights[i] for i in indices) for c,indices in classes.items()}
        total=sum(weights);remaining=count-len(classes)
        indices=[self.rng.choices(pool,weights=[weights[i] for i in pool],k=1)[0] for pool in classes.values()]
        indices+=self.rng.choices(range(len(entries)),weights=weights,k=remaining)
        self.rng.shuffle(indices)
        self.last_selected=[entries[i] for i in indices]
        self.last_sample_distribution={c:1/count+remaining/count*m/total for c,m in mass.items()}
        self.last_rehearsal=dict(retention=self.last_retention,cache_entries=len(entries),
            mean_factor=sum(factors)/len(factors),min_factor=min(factors),max_factor=max(factors),
            class_probabilities=self.last_sample_distribution,
            class_draws=dict(Counter(entries[i]['goal']['task_type'] for i in indices)),
            source_credit_total=sum(base),raw_quality_total=sum(e['goal']['hsl_weight'] for e in entries),
            mean_recent_feedback_sources=sum(sum(self.round_index-r<=10 for r in self.feedback_sources.get(family(e),{}).values()) for e in entries)/len(entries))
        return [dict(rollout=entries[i]['rollout'],goal=entries[i]['goal'],
            sample_probability=(weights[i]/mass[entries[i]['goal']['task_type']]+remaining*weights[i]/total)/count,
            cache_source_id=entries[i].get('cache_source_id'),policy_version=entries[i]['rollout']['policy_version'],
            collection_round=entries[i]['collection_round']) for i in indices]

    def observe_training(self,metrics):
        result=super().observe_training(metrics)
        for e in self.last_selected:
            self.feedback_sources.setdefault(family(e),{})[source_key(e)]=self.round_index
        self.feedback_sources={k:{s:r for s,r in v.items() if self.round_index-r<=10}
                               for k,v in self.feedback_sources.items()}
        self.feedback_sources={k:v for k,v in self.feedback_sources.items() if v}
        return result

    def state_dict(self):
        return dict(super().state_dict(),feedback_sources=self.feedback_sources,last_retention=self.last_retention)

    @classmethod
    def from_state_dict(cls,state):
        b=super().from_state_dict(state)
        b.feedback_sources=state.get('feedback_sources',{})
        b.last_retention=state.get('last_retention',{})
        return b
