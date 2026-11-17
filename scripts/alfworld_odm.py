"""ODM Algorithm 1 adapted to HSL's goal-grouped trajectory draws.

Source: https://arxiv.org/html/2312.02406 (Algorithm 1).
Each draw is one variable-length supervised trajectory; its mean token NLL
is a microbatch reward. Sum rewards per group, importance-correct by the actual
group probability, then EMA. HSL weighting remains within each group.
Alpha=.9 is an explicitly chosen local setting, not an author-verified default.
"""
import math


def total(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    return sum(total(x) for x in value) if isinstance(value, (list, tuple)) else float(value)


class ODMController:
    def __init__(self, alpha=.9, warmup_steps=1):
        if not 0 <= alpha < 1:
            raise ValueError('EMA alpha must be in [0,1)')
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.estimated_rewards = {}

    def logits(self, buffer, step):
        masses = {}
        for entry in buffer.demonstrations:
            g = entry['goal']
            masses[g['task_type']] = masses.get(g['task_type'], 0.) + g['hsl_weight']
        if not masses or step <= self.warmup_steps:
            return {}
        k = len(masses)
        if k == 1:
            return {next(iter(masses)): 0.}
        exploration = min(1 / k, math.sqrt(math.log(k) / (k * step)))
        previous = min(1 / k, math.sqrt(math.log(k) / (k * max(1, step - 1))))
        scores = {c: previous * self.estimated_rewards.get(c, 0.) for c in masses}
        maximum = max(scores.values())
        weights = {c: math.exp(v - maximum) for c, v in scores.items()}
        denominator = sum(weights.values())
        probabilities = {c: (1 - k * exploration) * weights[c] / denominator + exploration for c in masses}
        # Buffer samples w_HSL(d)*exp(logit_c); divide out group mass to
        # implement ODM's group distribution exactly, preserving within-group HSL.
        return {c: math.log(probabilities[c]) - math.log(masses[c]) for c in masses}

    def observe(self, step, provenance, probabilities, metrics):
        rewards = {}
        for row in provenance:
            if row['source'] != 'hindsight':
                continue
            draw = row['sampling_draw_id']
            tokens = total(metrics[f'odm_tokens_{draw}'])
            if tokens <= 0:
                raise ValueError('Sampled hindsight trajectory has no reward tokens')
            reward = total(metrics[f'odm_nll_{draw}']) / tokens
            if not math.isfinite(reward) or reward < 0:
                raise ValueError('Invalid ODM reward')
            c = row['goal_task_type']
            rewards[c] = rewards.get(c, 0.) + reward
        for c, reward in rewards.items():
            self.estimated_rewards[c] = self.alpha * self.estimated_rewards.get(c, 0.) + (1 - self.alpha) * reward / probabilities[c]
        return dict(step=step, group_reward_sums=rewards, probabilities=probabilities,
                    estimated_rewards=dict(self.estimated_rewards), alpha=self.alpha)

    def state_dict(self):
        return dict(alpha=self.alpha, warmup_steps=self.warmup_steps, estimated_rewards=self.estimated_rewards)

    @classmethod
    def from_state_dict(cls, state):
        result = cls(state['alpha'], state['warmup_steps'])
        result.estimated_rewards = dict(state['estimated_rewards'])
        return result
