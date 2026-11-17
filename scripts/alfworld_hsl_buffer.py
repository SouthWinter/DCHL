"""Versioned FIFO rollout buffer and trajectory-level HSL sampling.

Capacity counts relabeled goal/prefix demonstrations, as the paper defines D'.
Training-loop integration owns actor export and service reload; a version tag
here records that handoff but cannot prove the service loaded those weights.
"""
from collections import deque
import random

from alfworld_hindsight import prefix_messages
from alfworld_hsl_credit import tilted_weights


def training_records(rollout, goal, mixture=1.0, sampled_trajectories=1):
    if 'physical_context_projection' in rollout:
        from alfworld_cycle_training_context import training_records as physical_records
        return physical_records(rollout, goal, mixture, sampled_trajectories)
    if rollout.get('dialogue_hsl',False):
        from alfworld_dialogue_hsl import hindsight_records
        return hindsight_records(rollout,goal,mixture,sampled_trajectories)
    records = prefix_messages(rollout['steps'][:goal['prefix_length']], goal['instruction'])
    mask = goal['action_loss_mask']
    if len(mask) != len(records):
        raise ValueError('action mask length differs from executed prefix')
    for record, supervised in zip(records, mask):
        for message in record['messages']:
            message['loss_on_this_turn'] = message['role'] == 'assistant' and supervised
        record.update(enable_thinking=False, policy_version=rollout['policy_version'],
            source_episode_id=rollout['gamefile'], goal_task_type=goal['task_type'],
            action_supervised=supervised,
            # Multiply each action's SUM token NLL by this coefficient.
            # Applying this to an already token-averaged loss changes the objective.
            trajectory_action_coefficient=mixture / sampled_trajectories / goal['prefix_length'])
    return records


class HSLBuffer:
    def __init__(self, capacity=100, seed=42, allow_engineering=False):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.demonstrations = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        self.allow_engineering = allow_engineering
        self.round_index = 0
        self.current_policy_version = None
        self.last_sample_distribution = {}

    def append_round(self, results, policy_version):
        """Append completed relabels from one synchronized collection round."""
        if not policy_version:
            raise ValueError('collection round requires a policy version')
        for result in results:
            if result.get('engineering_fixture_only') and not self.allow_engineering:
                raise ValueError('historical engineering fixture cannot enter formal buffer')
            if result['policy_version'] != policy_version:
                raise ValueError('mixed policy versions in one collection round')
            if not result.get('action_relevance_ready'):
                raise ValueError('action relevance is missing')
        # Validate the complete round before mutating the buffer.
        for result in results:
            for goal in result['goals']:
                self.demonstrations.append(dict(rollout=result, goal=goal,
                    collection_round=self.round_index))
        self.current_policy_version = policy_version
        self.round_index += 1

    def sample(self, count, goal_logits=None):
        entries = list(self.demonstrations)
        self.last_sample_distribution = {}
        if not entries:
            return []  # Caller can perform the original-demonstration update alone.
        classes = [entry['goal']['task_type'] for entry in entries]
        weights = tilted_weights([entry['goal']['hsl_weight'] for entry in entries], classes, goal_logits)
        total = sum(weights)
        for kind, weight in zip(classes, weights):
            self.last_sample_distribution[kind] = self.last_sample_distribution.get(kind, 0.) + weight / total
        picked = self.rng.choices(range(len(entries)), weights=weights, k=count)
        return [dict(rollout=entries[i]['rollout'], goal=entries[i]['goal'],
                     sample_probability=weights[i] / total,
                     cache_source_id=entries[i].get('cache_source_id'),
                     policy_version=entries[i]['rollout']['policy_version'],
                     collection_round=entries[i]['collection_round']) for i in picked]

    def state_dict(self):
        return dict(capacity=self.capacity, demonstrations=list(self.demonstrations),
            rng_state=self.rng.getstate(), round_index=self.round_index,
            current_policy_version=self.current_policy_version,
            allow_engineering=self.allow_engineering)

    @classmethod
    def from_state_dict(cls, state):
        def tuples(value):
            return tuple(map(tuples, value)) if isinstance(value, (list, tuple)) else value
        buffer = cls(state['capacity'], allow_engineering=state['allow_engineering'])
        buffer.demonstrations.extend(state['demonstrations'])
        buffer.rng.setstate(tuples(state['rng_state']))
        buffer.round_index = state['round_index']
        buffer.current_policy_version = state['current_policy_version']
        return buffer
