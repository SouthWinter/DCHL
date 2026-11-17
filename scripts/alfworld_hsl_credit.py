"""Score-function credit for randomized goal-class sampling.

For p_beta(d) proportional to w_HSL(d) exp(beta[class(d)]), a batch of m
independent draws has score count[class] - m * P_beta(class). A fresh
post-update rollout reward times this score estimates the derivative of the
conditional ONE-UPDATE return. This is the standard likelihood-ratio identity,
not a per-trajectory causal value or a full-training meta-gradient.
"""
from collections import Counter
import math


def tilted_weights(base_weights, classes, logits=None):
    if not logits:
        return list(base_weights)
    offset = max(logits.get(kind, 0.) for kind in classes)
    return [weight * math.exp(logits.get(kind, 0.) - offset)
            for weight, kind in zip(base_weights, classes)]


class GoalSamplingController:
    """One-step delayed sampler learning; agent weights still use ordinary SFT."""
    def __init__(self, learning_rate=1 / math.sqrt(32)):
        self.learning_rate = learning_rate
        self.logits = {}
        self.pending = None

    def observe(self, collection_step, reward):
        if self.pending is None:
            return None
        if collection_step != self.pending['update_step'] + 1:
            raise ValueError('sampler feedback must follow the pending model update')
        p = self.pending
        credit = sampling_credit(p['counts'], p['probabilities'], reward, p['baseline'])
        for kind, gradient in credit['gradient'].items():
            self.logits[kind] = self.logits.get(kind, 0.) + self.learning_rate * gradient
        credit.update(update_step=p['update_step'], feedback_collection_step=collection_step)
        self.pending = None
        return credit

    def remember(self, update_step, reward, provenance, probabilities):
        if self.pending is not None:
            raise ValueError('previous update has not received its rollout feedback')
        counts = Counter(r['goal_task_type'] for r in provenance if r['source'] == 'hindsight')
        if counts:
            self.pending = dict(update_step=update_step, baseline=reward,
                                counts=dict(counts), probabilities=dict(probabilities))
        return dict(class_logits=dict(self.logits), class_probabilities=dict(probabilities),
                    sampled_counts=dict(counts), learning_rate=self.learning_rate)

    def state_dict(self):
        return dict(learning_rate=self.learning_rate, logits=self.logits, pending=self.pending)

    @classmethod
    def from_state_dict(cls, state):
        controller = cls(state['learning_rate'])
        controller.logits = dict(state['logits'])
        controller.pending = state['pending']
        return controller


def sampling_credit(counts, probabilities, reward, baseline):
    size = sum(counts.values())
    classes = sorted(set(counts) | set(probabilities))
    score = {kind: counts.get(kind, 0) - size * probabilities.get(kind, 0.) for kind in classes}
    return dict(batch_size=size, before_reward=baseline, after_reward=reward,
        reward_change=reward - baseline, score=score,
        gradient={kind: (reward - baseline) * score[kind] for kind in classes})


def completed_transition_credits(rounds):
    credits = []
    for before, after in zip(rounds, rounds[1:]):
        if after['step'] != before['step'] + 1:
            continue
        if not before['sampled_goals']:
            continue
        value = sampling_credit(before['sampled_goal_counts'], before['expected_goal_distribution'],
            after['successes'] / after['rollout_count'], before['successes'] / before['rollout_count'])
        value.update(update_step=before['step'], feedback_collection_step=after['step'],
                     collection_policy_version=after['policy_version'])
        credits.append(value)
    return dict(
        interpretation='local goal-logit gradient estimates along the observed online model path',
        transitions=credits,
        caveats=['This analysis applies no sampler changes to the observed run.',
                 'Fresh uniformly sampled training tasks estimate the TRAINING-pool stochastic return.',
                 'The pre-update reward is a baseline, not a matched no-hindsight counterfactual.',
                 'Success fluctuation alone gives no class credit; centered random class counts are required.',
                 'Few transitions have high variance; their signs cannot certify a useful data class.',
                 'The last update has no next-round feedback unless another collection is performed.'])
