"""Render verified continuation labels on the actual executed history.

Collection can search with a shortened history. Learning uses the physical
prefix at each retained action; masks, draw weights and compact denominator
stay unchanged. Removed loop actions are input only, never additional targets.
"""
import copy


def attach_physical_context(row):
    if not row['cycle_resample']['triggered']:
        return
    segment = row['hindsight_segments'][1]
    actor = row['actor_trajectory']
    physical = row['trajectory']
    positions = [s['physical_step'] for s in actor]
    start, cut = row['cycle_resample']['start'], row['cycle_resample']['cut']
    assert positions == [i if i < start else i + cut - start for i in range(len(actor))]
    assert len(segment['steps']) == len(positions)
    for saved, position in zip(segment['steps'], positions):
        original = physical[position]
        assert all(saved[k] == original[k] for k in
                   ('action', 'observation', 'result_observation', 'admissible_actions'))
    segment['physical_context_projection'] = dict(
        positions=positions, steps=physical, start=start, cut=cut,
        collection_context='shortened', training_context='physical',
        extra_environment_actions=0, extra_targets=0)


def training_records(rollout, goal, mixture, count):
    from alfworld_dialogue_hsl import hindsight_records
    projection = rollout['physical_context_projection']
    length = goal['prefix_length']
    positions = projection['positions'][:length]
    assert len(positions) == length and length > projection['start']
    full_length = positions[-1] + 1
    physical_goal = dict(goal, prefix_length=full_length,
                         action_loss_mask=[False] * full_length)
    for position, supervised in zip(positions, goal['action_loss_mask']):
        physical_goal['action_loss_mask'][position] = supervised
    physical_rollout = dict(rollout, steps=projection['steps'][:full_length])
    all_records = hindsight_records(physical_rollout, physical_goal, mixture, count)
    compact = hindsight_records(rollout, goal, mixture, count)
    records = []
    for index, position in enumerate(positions):
        record = copy.deepcopy(all_records[position])
        before = compact[index]
        assert record['action'] == before['action']
        assert record['messages'][-1] == before['messages'][-1]
        assert record['action_supervised'] == before['action_supervised']
        record.update(step=index, physical_step=position,
                      trajectory_action_coefficient=before['trajectory_action_coefficient'],
                      training_context='physical')
        records.append(record)
    assert len(records) == length
    return records
