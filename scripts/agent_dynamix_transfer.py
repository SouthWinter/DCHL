"""DynamixSFT's arm policy on verified PlanCraft and WebShop goals."""
import math
import random
from collections import Counter, defaultdict

import alfworld_batch_selection as selection
from alfworld_stable_evidence import source_key


WEB_SHOP_FAMILIES = (
    "women's socks & hosiery", 'hair loss products', 'toothpicks',
    'video surveillance', 'binoculars & scopes', "women's shorts",
    "men's henleys", 'makeup palettes',
)


class DynamixTransferBuffer(selection.SelectionBuffer):
    alpha, beta, gamma = .8, 4., .2
    dataset = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm_values = {}
        self.reward_updates = 0
        self.last_dynamix = {}

    def arm(self, goal):
        if self.dataset == 'plancraft':
            n = goal['relevant_actions']
            return 'one_decision' if n == 1 else 'two_decisions' if n == 2 else 'three_plus_decisions'
        if self.dataset == 'webshop':
            family = goal['object_type']
            return family if family in WEB_SHOP_FAMILIES else 'other_families'
        raise ValueError('Unknown dataset')

    def candidate_goals(self):
        retained = {source_key(e): e['collection_round'] for e in self.demonstrations}
        groups = defaultdict(list)
        for key in sorted(self.raw_sources):
            if key not in retained:
                continue
            rollout = self.raw_sources[key]
            for goal in rollout['goals']:
                groups[self.arm(goal)].append(dict(rollout=rollout, goal=goal,
                                                  collection_round=retained[key]))
        return dict(groups)

    def arm_distribution(self, groups):
        names = sorted(k for k, v in groups.items() if v)
        if not names:
            return {}
        total = sum(len(groups[k]) for k in names)
        prior = {k: len(groups[k]) / total for k in names}
        weights = {k: prior[k] * math.exp(self.beta * self.arm_values.get(k, 0.)) for k in names}
        denominator = sum(weights.values())
        return {k: (1-self.gamma)*weights[k]/denominator+self.gamma/len(names) for k in names}

    def rows_for(self, entry):
        return [r for r in selection.training_records(entry['rollout'], entry['goal'], .3, 32)
                if r['action_supervised']]

    def propose(self, package, step, seed):
        groups = self.candidate_goals()
        probabilities = self.arm_distribution(groups)
        if not probabilities:
            return None
        rng = random.Random(seed+step*7919+101)
        names = sorted(probabilities)
        chosen = [rng.choice(groups[rng.choices(names,
                  weights=[probabilities[k] for k in names], k=1)[0]]) for _ in range(32)]
        aux_count = sum(p['actions'] for p in package['provenance']
                        if p['source'] == 'verified_recovery')
        cut = len(package['rows'])-aux_count
        original = [r for r in package['rows'][:cut] if 'policy_version' not in r]
        auxiliary = package['rows'][cut:]
        assert original and all('policy_version' in r for r in auxiliary)
        rows = list(original)
        provenance = [p for p in package['provenance'] if p['source'] == 'original']
        for draw, entry in enumerate(chosen):
            goal, rollout = entry['goal'], entry['rollout']
            records = self.rows_for(entry)
            if not records:
                raise ValueError('Verified goal without supervised actions')
            feedback = set(goal['feedback_steps'])
            for row in records:
                row['sampling_draw_id'] = draw if row['step'] in feedback else -1
            if not any(r['sampling_draw_id'] == draw for r in records):
                raise ValueError('Verified goal without feedback action')
            rows.extend(records)
            provenance.append(dict(source='hindsight', task=rollout['gamefile'],
                sampling_draw_id=draw, goal_arm=self.arm(goal), goal=goal['instruction'],
                prefix_length=goal['prefix_length'], supervised_actions=goal['relevant_actions'],
                feedback_steps=goal['feedback_steps'],
                feedback_actions=sum(r['sampling_draw_id'] == draw for r in records),
                policy_version=rollout['policy_version'], cache_source_id=None,
                allocation='DynamixSFT arm draw, uniform verified goal within arm'))
        rng.shuffle(rows)
        rows.extend(auxiliary)
        provenance.extend(p for p in package['provenance'] if p['source'] == 'verified_recovery')
        return dict(rows=rows, provenance=provenance, selected=chosen,
                    distribution=probabilities,
                    candidate_counts={k: len(v) for k, v in groups.items()})

    def probe_rows(self, step, seed):
        groups = self.candidate_goals()
        rng = random.Random(seed+step*15401+707)
        rows = []
        for arm in sorted(groups):
            actions = self.rows_for(rng.choice(groups[arm]))
            if not actions:
                raise ValueError('Probe goal without supervised action')
            for row in rng.sample(actions, min(4, len(actions))):
                row['dynamix_arm'] = arm
                row['sampling_draw_id'] = -1
                rows.append(row)
        return rows, sorted(groups)

    def observe_probe(self, before, after):
        assert before.keys() == after.keys() and before
        raw = {k: (before[k]-after[k])/(before[k]+1e-8) for k in before}
        lo, hi = min(raw.values()), max(raw.values())
        norm = {k: (v-lo)/(hi-lo) if hi>lo+1e-12 else .5 for k, v in raw.items()}
        for k, v in norm.items():
            self.arm_values[k] = self.alpha*self.arm_values.get(k, 0.)+(1-self.alpha)*v
        self.reward_updates += 1
        return dict(before=before, after=after, relative_improvement=raw,
                    normalized_reward=norm, arm_values=dict(self.arm_values),
                    reward_updates=self.reward_updates)

    def commit_proposal(self, proposal):
        super().commit_proposal(proposal)
        self.last_dynamix = dict(arm_probabilities=proposal['distribution'],
            candidate_counts=proposal['candidate_counts'],
            draws=dict(Counter(self.arm(e['goal']) for e in proposal['selected'])),
            arm_values=dict(self.arm_values))

    def state_dict(self):
        return dict(super().state_dict(), dynamix_arm_values=self.arm_values,
                    dynamix_reward_updates=self.reward_updates,
                    dynamix_last=self.last_dynamix)

    @classmethod
    def from_state_dict(cls, state):
        buffer = super().from_state_dict(state)
        buffer.arm_values = state.get('dynamix_arm_values', {})
        buffer.reward_updates = state.get('dynamix_reward_updates', 0)
        buffer.last_dynamix = state.get('dynamix_last', {})
        return buffer
