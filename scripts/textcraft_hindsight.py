"""Complete TRAIN goals verified in the source TextCraft recipe context."""
from textcraft_adapter import action_records, native_support


def relabel(example, trajectory, tokenizer, version, session, targets):
    actions = [step['action'] for step in trajectory]
    candidates = list(dict.fromkeys(step['observation'].split()[-1] for step in trajectory
        if step['observation'].startswith('Crafted ') and step['observation'].split()[-1] in targets))
    goals = []
    checks = 0
    for goal in candidates:
        supported = native_support(session, example, actions, goal)
        assert supported is not None
        replay = supported['full']
        length = len(replay['trajectory'])
        checks += length + 2
        assert all(a['action'] == b['action'] and a['observation'] == b['observation']
            for a, b in zip(trajectory[:length], replay['trajectory']))
        kept = supported['feedback_steps']
        records = action_records(replay['messages'][:-1], tokenizer)
        assert len(records) == length
        n = len(kept)
        goals.append(dict(instruction=replay['description'], task_type='craft', object_type=goal,
            native_goal=goal, prefix_length=length, relevant_actions=n,
            action_loss_mask=[i in kept for i in range(length)], feedback_steps=kept,
            hsl_weight=(n / length)**.8*n, records=records))
    return dict(gamefile=example['id'], policy_version=version, action_relevance_ready=True,
        requested_task_type='craft', source_success=bool(trajectory and trajectory[-1]['won']),
        goals=goals, candidate_count=len(candidates), relevance_checks=checks,
        label_source='Actually crafted TRAIN goals; full native success; unchanged recipe context and feedback; native reverse deletion')
