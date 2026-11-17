"""Original FIFO HSL with the DCHL verified continuation channel only."""
import random
from collections import Counter
from alfworld_hsl_buffer import HSLBuffer, training_records
from alfworld_hsl_batch import mixed_records as original_records


class HSLRecoveryBuffer(HSLBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recoveries = []
        self.recovery_demand = []

    def append_round(self, results, policy_version):
        extra = [e for r in results for e in r.get('_verified_recoveries', [])]
        clean = [{k: v for k, v in r.items() if k != '_verified_recoveries'} for r in results]
        super().append_round(clean, policy_version)
        self.recovery_demand.extend(dict(round=self.round_index, task=r['requested_task_type'],
                                        failed=not r['source_success']) for r in clean)
        self.recovery_demand = [r for r in self.recovery_demand if self.round_index-r['round'] < 10]
        pool = {r['key']: r for r in self.recoveries}
        for r in extra:
            pool.setdefault(r['key'], r)
        self.recoveries = sorted((r for r in pool.values() if self.round_index-r['round'] < 10),
                                 key=lambda r: (r['round'], r['key']))[-128:]

    def state_dict(self):
        return dict(super().state_dict(), recovery_bank=self.recoveries,
                    recovery_demand=self.recovery_demand, intervention='hsl_plus_recovery_v1')

    @classmethod
    def from_state_dict(cls, state):
        if state.get('intervention') != 'hsl_plus_recovery_v1':
            raise ValueError('Cannot resume a different intervention')
        b = super().from_state_dict(state)
        b.recoveries = state['recovery_bank']
        b.recovery_demand = state['recovery_demand']
        return b


def mixed_records(demonstrations, buffer, rng, **kwargs):
    rows, provenance = original_records(demonstrations, buffer, rng, **kwargs)
    if not buffer.recoveries:
        return rows, provenance
    attempts = Counter(r['task'] for r in buffer.recovery_demand)
    failures = Counter(r['task'] for r in buffer.recovery_demand if r['failed'])
    groups = {}
    for e in buffer.recoveries:
        groups.setdefault(e['goal']['task_type'], []).append(e)
    aux = random.Random(991003+buffer.round_index)
    keys = sorted(groups)
    chosen = [aux.choice(groups[task]) for task in aux.choices(
        keys, weights=[(failures[k]+1)/(attempts[k]+4) for k in keys], k=4)]
    for e in chosen:
        records = [r for r in training_records(e['rollout'], e['goal']) if r['action_supervised']]
        for r in records:
            r.update(trajectory_action_coefficient=.05/len(chosen)/len(records), sampling_draw_id=-1)
        rows.extend(records)
        provenance.append(dict(source='verified_recovery', key=e['key'], cut=e['cut'],
                               actions=len(records), coefficient=.05/len(chosen)))
    return rows, provenance
