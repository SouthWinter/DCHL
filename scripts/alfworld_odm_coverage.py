"""Keep ODM class probabilities; cover distinct demonstrations within each batch."""
from collections import defaultdict
from alfworld_hsl_buffer import HSLBuffer
from alfworld_hsl_credit import tilted_weights


def demonstration_key(entry):
    r,g=entry['rollout'],entry['goal']
    return (r['policy_version'],r['gamefile'],g['task_type'],g['instruction'],g['prefix_length'])


class CoverageBuffer(HSLBuffer):
    coverage_key = staticmethod(demonstration_key)

    def sample(self,count,goal_logits=None,*,class_sequence=None):
        entries=list(self.demonstrations)
        self.last_sample_distribution={};self.last_coverage_samples=[]
        if not entries:return []
        classes=[e['goal']['task_type'] for e in entries]
        weights=tilted_weights([e['goal']['hsl_weight'] for e in entries],classes,goal_logits)
        grouped=defaultdict(list);mass=defaultdict(float)
        for i,(c,w) in enumerate(zip(classes,weights)):
            grouped[c].append(i);mass[c]+=w
        total=sum(mass.values())
        self.last_sample_distribution={c:v/total for c,v in mass.items()}
        order=list(grouped)
        if class_sequence is None:
            class_sequence=self.rng.choices(order,weights=[mass[c] for c in order],k=count)
        if len(class_sequence)!=count or any(c not in grouped for c in class_sequence):
            raise ValueError('Invalid coverage class sequence')
        available={c:list(indices) for c,indices in grouped.items()};cycles=defaultdict(int)
        result=[]
        for c in class_sequence:
            if not available[c]:
                available[c]=list(grouped[c]);cycles[c]+=1
            pool=available[c];denominator=sum(weights[i] for i in pool)
            i=self.rng.choices(pool,weights=[weights[j] for j in pool],k=1)[0]
            entry=entries[i];key=demonstration_key(entry)
            conditional=sum(weights[j] for j in pool if demonstration_key(entries[j])==key)/denominator
            coverage_key=self.coverage_key(entry)
            available[c]=[j for j in pool if self.coverage_key(entries[j])!=coverage_key]
            result.append(dict(rollout=entry['rollout'],goal=entry['goal'],
                sample_probability=self.last_sample_distribution[c]*conditional,
                cache_source_id=entry.get('cache_source_id'),policy_version=entry['rollout']['policy_version'],
                collection_round=entry['collection_round'],coverage_class_probability=self.last_sample_distribution[c],
                coverage_conditional_probability=conditional,coverage_cycle=cycles[c]))
        self.last_coverage_samples=result
        return result


def coverage_records(*args,**kwargs):
    from alfworld_hsl_batch import mixed_records
    rows,provenance=mixed_records(*args,**kwargs)
    buffer=args[1]
    for r in provenance:
        if r['source']=='hindsight':
            sample=buffer.last_coverage_samples[r['sampling_draw_id']]
            r.update({k:sample[k] for k in ('coverage_class_probability','coverage_conditional_probability','coverage_cycle')})
    return rows,provenance
