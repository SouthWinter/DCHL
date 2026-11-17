"""DynamixSFT's dataset-arm policy over verified ALFWorld goal families.

The paper's datasets become the six public task/goal families. Candidate goals,
supervision and verified original-task continuations use the DCHL carrier.
"""
import math
import random
from collections import Counter, defaultdict

from alfworld_batch_selection import SelectionBuffer
from alfworld_hsl_buffer import training_records
from alfworld_stable_evidence import source_key


class DynamixSFTBuffer(SelectionBuffer):
    feedback_metric_key = 'dynamixsft'
    alpha = 0.8
    beta = 4.0
    gamma = 0.2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm_values = {}
        self.reward_updates = 0
        self.last_dynamix = {}

    def candidate_goals(self):
        retained = {source_key(e): e['collection_round'] for e in self.demonstrations}
        groups = defaultdict(list)
        for key in sorted(self.raw_sources):
            if key not in retained:
                continue
            rollout = self.raw_sources[key]
            for goal in rollout['goals']:
                groups[goal['task_type']].append(dict(
                    rollout=rollout, goal=goal, collection_round=retained[key]))
        return dict(groups)

    def arm_distribution(self, groups):
        names = sorted(k for k, v in groups.items() if v)
        if not names:
            return {}
        count = sum(len(groups[k]) for k in names)
        prior = {k: len(groups[k]) / count for k in names}
        unnormalized = {k: prior[k] * math.exp(self.beta * self.arm_values.get(k, 0.0))
                        for k in names}
        denominator = sum(unnormalized.values())
        return {k: (1 - self.gamma) * unnormalized[k] / denominator
                + self.gamma / len(names) for k in names}

    def propose(self, package, step, seed):
        groups = self.candidate_goals()
        probabilities = self.arm_distribution(groups)
        if not probabilities:
            return None
        rng = random.Random(seed + step * 7919 + 101)
        names = sorted(probabilities)
        chosen = [rng.choice(groups[rng.choices(names,
                    weights=[probabilities[k] for k in names], k=1)[0]])
                  for _ in range(32)]
        aux_count = sum(p['actions'] for p in package['provenance']
                        if p['source'] == 'verified_recovery')
        original = [r for r in package['rows'][:len(package['rows']) - aux_count]
                    if 'policy_version' not in r]
        auxiliary = package['rows'][len(package['rows']) - aux_count:]
        assert original and all('policy_version' in r for r in auxiliary)
        rows = list(original)
        provenance = [p for p in package['provenance'] if p['source'] == 'original']
        for draw, entry in enumerate(chosen):
            goal, rollout = entry['goal'], entry['rollout']
            records = [r for r in training_records(rollout, goal, .3, 32)
                       if r['action_supervised']]
            if not records:
                raise ValueError('Verified candidate without supervised action')
            feedback = set(goal['feedback_steps'])
            for row in records:
                row['sampling_draw_id'] = draw if row['step'] in feedback else -1
            if not any(r['sampling_draw_id'] == draw for r in records):
                raise ValueError('Verified candidate without feedback action')
            rows.extend(records)
            provenance.append(dict(source='hindsight', task=rollout['gamefile'],
                sampling_draw_id=draw, goal_task_type=goal['task_type'],
                goal=goal['instruction'], prefix_length=goal['prefix_length'],
                supervised_actions=goal['relevant_actions'],
                feedback_steps=goal['feedback_steps'],
                feedback_actions=sum(r['sampling_draw_id'] == draw for r in records),
                policy_version=rollout['policy_version'], cache_source_id=None,
                allocation='DynamixSFT task-family arm, uniform verified goal within arm'))
        rng.shuffle(rows)
        rows.extend(auxiliary)
        provenance.extend(p for p in package['provenance']
                          if p['source'] == 'verified_recovery')
        return dict(rows=rows, provenance=provenance, selected=chosen,
                    distribution=probabilities, candidate_counts={k: len(v) for k, v in groups.items()})

    def probe_rows(self, step, seed):
        groups = self.candidate_goals()
        rng = random.Random(seed + step * 15401 + 707)
        rows = []
        for name in sorted(groups):
            entry = rng.choice(groups[name])
            actions = [r for r in training_records(entry['rollout'], entry['goal'], .3, 32)
                       if r['action_supervised']]
            if not actions:
                raise ValueError('Probe candidate without supervised action')
            for row in rng.sample(actions, min(4, len(actions))):
                row['dynamix_arm'] = name
                row['sampling_draw_id'] = -1
                rows.append(row)
        return rows, sorted(groups)

    def observe_probe(self, before, after):
        assert before.keys() == after.keys() and before
        raw = {k: (before[k] - after[k]) / (before[k] + 1e-8) for k in before}
        lo, hi = min(raw.values()), max(raw.values())
        normalized = {k: (v - lo) / (hi - lo) if hi > lo + 1e-12 else 0.5
                      for k, v in raw.items()}
        for k, value in normalized.items():
            self.arm_values[k] = self.alpha * self.arm_values.get(k, 0.0) + (1 - self.alpha) * value
        self.reward_updates += 1
        return dict(before=before, after=after, relative_improvement=raw,
                    normalized_reward=normalized, arm_values=dict(self.arm_values),
                    reward_updates=self.reward_updates)

    def commit_proposal(self, proposal):
        super().commit_proposal(proposal)
        self.last_dynamix = dict(arm_probabilities=proposal['distribution'],
            candidate_counts=proposal['candidate_counts'],
            draws=dict(Counter(e['goal']['task_type'] for e in proposal['selected'])),
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
