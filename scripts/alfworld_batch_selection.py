"""Alternative source/goal allocation within execution-valid trajectories."""
import copy
import random
from collections import defaultdict, Counter
from alfworld_verified_recovery import VerifiedRecoveryBuffer
from alfworld_stable_evidence import source_key
from alfworld_hsl_buffer import training_records


def scene(text):
    text = text.split('Your task is to:')[0]
    if 'You are in' not in text:
        raise ValueError('Missing initial scene in action-contract context')
    return ' '.join(text[text.index('You are in'):].split())


class SelectionBuffer(VerifiedRecoveryBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_sources = {}

    def append_round(self, results, policy_version):
        for r in results:
            if r['goals']:
                clean = {k:v for k,v in r.items() if k!='_verified_recoveries'}
                self.raw_sources[source_key(dict(rollout=clean))] = clean
        super().append_round(results, policy_version)
        retained = {source_key(e) for e in self.demonstrations}
        self.raw_sources = {k:v for k,v in self.raw_sources.items() if k in retained}

    def state_dict(self):
        return dict(super().state_dict(), raw_sources=self.raw_sources)

    @classmethod
    def from_state_dict(cls, state):
        b = super().from_state_dict(state)
        b.raw_sources = state.get('raw_sources', {})
        return b

    def goal_weights(self, candidates, chosen):
        return self.base_weights(candidates)

    def propose(self, package, step, seed):
        entries = list(self.demonstrations)
        if not entries:
            return None
        rng = random.Random(seed+step*7919+101)
        weights = [b*self.factor(e) for b,e in zip(self.base_weights(entries), entries)]
        groups = defaultdict(list)
        classes = defaultdict(list)
        for i,e in enumerate(entries):
            groups[source_key(e)].append(i)
            classes[e['goal']['task_type']].append(i)
        # Keep one existing maximal certificate from every supported class.
        chosen = [entries[rng.choices(pool, weights=[weights[i] for i in pool], k=1)[0]]
                  for pool in classes.values()]
        keys = sorted(groups)
        masses = [sum(weights[i] for i in groups[k]) for k in keys]
        for key in rng.choices(keys, weights=masses, k=32-len(chosen)):
            rollout = self.raw_sources[key]
            candidates = [dict(rollout=rollout, goal=g, collection_round=entries[groups[key][0]]['collection_round'])
                          for g in rollout['goals']]
            chosen.append(rng.choices(candidates, weights=self.goal_weights(candidates, chosen), k=1)[0])
        rng.shuffle(chosen)
        aux_count = sum(p['actions'] for p in package['provenance'] if p['source']=='verified_recovery')
        cut = len(package['rows'])-aux_count
        original = [r for r in package['rows'][:cut] if 'policy_version' not in r]
        auxiliary = package['rows'][cut:]
        assert original and all('policy_version' in r for r in auxiliary)
        rows, provenance = list(original), [p for p in package['provenance'] if p['source']=='original']
        for draw,e in enumerate(chosen):
            g, rollout = e['goal'], e['rollout']
            records = [r for r in training_records(rollout, g, .3, 32) if r['action_supervised']]
            loss_steps = g.get('loss_feedback_steps', g['feedback_steps'])
            for r in records:
                r['sampling_draw_id'] = draw if r['step'] in loss_steps else -1
            rows.extend(records)
            provenance.append(dict(source='hindsight', task=rollout['gamefile'], sampling_draw_id=draw,
                goal_task_type=g['task_type'], goal=g['instruction'], prefix_length=g['prefix_length'],
                supervised_actions=g['relevant_actions'], feedback_steps=g['feedback_steps'],
                feedback_actions=sum(r['sampling_draw_id']>=0 for r in records),
                policy_version=rollout['policy_version'], cache_source_id=None,
                allocation='source-shared quality then valid goal; class coverage retained'))
            if 'loss_feedback_steps' in g:
                provenance[-1]['loss_feedback_steps'] = loss_steps
        rng.shuffle(rows)
        rows.extend(auxiliary)
        provenance.extend(p for p in package['provenance'] if p['source']=='verified_recovery')
        return dict(rows=rows, provenance=provenance, selected=chosen,
                    distribution={c:n/32 for c,n in Counter(e['goal']['task_type'] for e in chosen).items()})

    def commit_proposal(self, proposal):
        self.last_selected = proposal['selected']
        self.last_sample_distribution = proposal['distribution']
        self.last_rehearsal = dict(self.last_rehearsal, committed_batch='B',
                                  class_draws=dict(Counter(e['goal']['task_type'] for e in self.last_selected)))


def reference_records(demonstrations, a, b, step, seed):
    excluded = {scene(r['messages'][2]['content']) for r in a['rows']+b['rows']}
    eligible = [k for k,v in demonstrations.items() if scene(v[0]['messages'][2]['content']) not in excluded]
    rng = random.Random(seed+step*15401+503)
    selected = rng.sample(sorted(eligible), min(16,len(eligible)))
    rows = []
    for key in selected:
        records = copy.deepcopy(demonstrations[key])
        for r in records:
            r['trajectory_action_coefficient'] = 1./len(selected)/len(records)
            r['sampling_draw_id'] = -1
        rows.extend(records)
    return rows, dict(tasks=selected, excluded_scenes=len(excluded), eligible_tasks=len(eligible))
