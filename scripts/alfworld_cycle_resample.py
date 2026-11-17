"""One exact-state history rollback within 40 actions.

The live environment is never reset. Both the discarded executed prefix's
achieved goals and the replayable pruned history's achieved goals are retained.
"""
import copy
import json
import os
import time


def cycle_start(states, observations, end):
    """Earliest online trigger, shortest qualifying loop at that end."""
    if not 2 <= end <= 20:
        return None
    for start in range(end-2, -1, -1):
        if states[start] != states[end] or observations[start] != observations[end]:
            continue
        seen = set(observations[:start+1])
        if all(observation in seen for observation in observations[start+1:end+1]):
            return start
    return None


def collect(row, config):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
    import alfworld_dialogue_hsl as dialogue
    from alfworld_hindsight import facts_of, replay_and_relabel
    began = time.monotonic()
    result = dict(row, trajectory=[], status='running', dialogue_hsl=True,
        policy_version=config['policy_version'], generation_usage=[])
    env = None
    try:
        with dialogue.urllib.request.urlopen(config['endpoint']+'/hsl/policy_version', timeout=30) as response:
            version = json.load(response)
        assert version['policy_version'] == config['policy_version'] and version['pending_version'] is None
        env = textworld.start(row['gamefile'], request_infos=textworld.EnvInfos(
            won=True, admissible_commands=True, facts=True), wrappers=[AlfredDemangler(shuffle=False)])
        state = env.reset()
        messages = dialogue.initial_messages(str(state.feedback))
        result['task_description'] = str(state.feedback).split('Your task is to: ')[-1].split('\n')[0]
        states, observations = [facts_of(state)], [str(state.feedback)]
        actor_steps, cut, start = [], None, None
        discarded_prefix = None
        for physical_step in range(config['max_steps']):
            observation, admissible = str(state.feedback), list(state.admissible_commands)
            prompt = dialogue.context_messages(config['endpoint'], messages)
            seed = config['seed'] + row['rollout_id']*1009 + physical_step
            output = dialogue.api(config['endpoint'], '/v1/chat/completions', dict(
                model='qwen3-4b', messages=prompt, temperature=1., top_p=1., max_tokens=512,
                seed=seed, chat_template_kwargs={'enable_thinking': False}))
            response = output['choices'][0]['message']['content'] or ''
            raw = dialogue.raw_action(response)
            action = dialogue.environment_action(raw, admissible) or 'invalid action'
            state, _, done = env.step(action)
            step = dict(step=physical_step, observation=observation, admissible_actions=admissible,
                action=action, raw_action=raw, response=response,
                result_observation=str(state.feedback), won=bool(state.won), done=bool(done))
            result['trajectory'].append(step)
            actor_steps.append(dict(step, step=len(actor_steps), physical_step=physical_step))
            result['generation_usage'].append(dict(step=physical_step, seed=seed, guided=False,
                prompt_tokens=int(output['usage']['prompt_tokens']),
                completion_tokens=int(output['usage']['completion_tokens']),
                history_actions_before_generation=len(actor_steps)-1))
            messages.extend([dict(role='assistant', content='Action: '+dialogue.public_action(action)),
                dict(role='user', content='Observation: '+str(state.feedback))])
            states.append(facts_of(state)); observations.append(str(state.feedback))
            if done or state.won:
                break
            if cut is None:
                match = cycle_start(states, observations, physical_step+1)
                if match is not None:
                    start, cut = match, physical_step+1
                    discarded_prefix = copy.deepcopy(result['trajectory'])
                    actor_steps = actor_steps[:start]
                    # Every retained message is a real earlier decision prefix.
                    messages = dialogue.initial_messages(result['trajectory'][0]['observation'])
                    for retained in actor_steps:
                        messages.extend([dict(role='assistant', content='Action: '+dialogue.public_action(retained['action'])),
                            dict(role='user', content='Observation: '+retained['result_observation'])])
                    assert states[start] == facts_of(state) and observations[start] == str(state.feedback)
        result.update(status='complete', won=bool(state.won), reward=float(state.won),
            steps=len(result['trajectory']), actor_trajectory=actor_steps)
        sources = []
        if discarded_prefix is not None:
            sources.append(dict(result, trajectory=discarded_prefix, won=False, reward=0., steps=cut))
        sources.append(dict(result, trajectory=actor_steps, steps=len(actor_steps)))
        segments = []
        prefix_goals = 0
        omitted_shared_goals = 0
        for index, source in enumerate(sources):
            segment = replay_and_relabel(source, config.get('relevance_rule', 'backward'))
            from alfworld_action_contract import mask_repeated_navigation
            segment['goals'] = [mask_repeated_navigation(segment['steps'], g) for g in segment['goals']]
            segment.update(attempt_index=index, source_attempt_id=f"{row['rollout_id']}:view-{index}",
                source_physical_attempt=0, cycle_resample=True, dialogue_hsl=True,
                engineering_fixture_only=False, response_contract='action_only_v1')
            if discarded_prefix is not None and index == 0:
                prefix_goals = len(segment['goals'])
            if discarded_prefix is not None and index == 1:
                # Shared pre-cycle achievements already have their exact original supervision.
                original_goals = segment['goals']
                segment['goals'] = [g for g in original_goals if g['prefix_length'] > start]
                omitted_shared_goals = len(original_goals)-len(segment['goals'])
            segments.append(segment)
        result.update(hindsight=segments[0], hindsight_segments=segments,
            cycle_resample=dict(triggered=cut is not None, start=start, cut=cut,
                removed_actions=cut-start if cut is not None else 0, physical_attempts=1, resets=0,
                neural_actions=result['steps'], prefix_goals=prefix_goals,
                continuation_goals=len(segments[-1]['goals']), omitted_shared_goals=omitted_shared_goals,
                teacher_calls=0, planner_calls=0, action_budget=config['max_steps'],
                history_state_and_observation_equal=True, progress_discarded=False),
            wall_seconds=time.monotonic()-began)
        if os.environ.get('ALFWORLD_CYCLE_PHYSICAL_CONTEXT') == '1':
            from alfworld_cycle_training_context import attach_physical_context
            attach_physical_context(result)
        return result
    except Exception as error:
        result.update(status='error', error=f'{type(error).__name__}: {error}', wall_seconds=time.monotonic()-began)
        return result
    finally:
        if env is not None:
            env.close()
