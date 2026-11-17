"""Allocate same-source goals by coverage of their execution decisions."""
from collections import Counter
from alfworld_batch_selection import SelectionBuffer
from alfworld_stable_evidence import source_key


class DecisionCoverageBuffer(SelectionBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_exposure = {}

    def append_round(self, results, policy_version):
        super().append_round(results, policy_version)
        self.decision_exposure = {k:v for k,v in self.decision_exposure.items() if k in self.raw_sources}

    def goal_weights(self, candidates, chosen):
        key = source_key(candidates[0])
        counts = Counter(self.decision_exposure.get(key, {}))
        # Include draws already reserved in this proposal, including coverage slots.
        for entry in chosen:
            if source_key(entry) == key:
                counts.update(str(j) for j in set(entry['goal']['feedback_steps']))
        frequency = Counter(j for e in candidates for j in set(e['goal']['feedback_steps']))
        return [e['goal']['hsl_weight'] * self.factor(e) *
                sum(1. / (frequency[j] * (1 + counts[str(j)]))
                    for j in set(e['goal']['feedback_steps'])) for e in candidates]

    def propose(self, package, step, seed):
        proposal = super().propose(package, step, seed)
        if proposal:
            for p in proposal['provenance']:
                if p['source'] == 'hindsight':
                    p['allocation'] = 'source budget; shared decision credit / (1 + committed and reserved exposure), times quality and bounded feedback'
        return proposal

    def commit_proposal(self, proposal):
        super().commit_proposal(proposal)
        for entry in proposal['selected']:
            key = source_key(entry)
            counts = self.decision_exposure.setdefault(key, {})
            for j in set(entry['goal']['feedback_steps']):
                counts[str(j)] = counts.get(str(j), 0) + 1

    def state_dict(self):
        return dict(super().state_dict(), decision_exposure=self.decision_exposure)

    @classmethod
    def from_state_dict(cls, state):
        b = super().from_state_dict(state)
        b.decision_exposure = state.get('decision_exposure', {})
        return b
