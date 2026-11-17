"""Preselected student prefixes, followed by same-student reference guidance."""
import copy
import random
import re
import json
import os
import select
import signal
import socket
import time
from pathlib import Path

HEADER = ('\nA successful example from another training episode follows. Adapt it '
          'to the current observations and original task; object identifiers and '
          'locations may differ.\n')


def query_words(text):
    return set(re.findall(r'[a-z]+',text.lower()))


def prepare_bank(bank, tokenizer, max_tokens=512):
    result = []
    for entry in bank:
        hint = HEADER + entry['text']
        tokens = len(tokenizer.encode(hint, add_special_tokens=False))
        if tokens <= max_tokens:
            result.append(dict(entry, hint=hint, hint_tokens=tokens))
    if not result:
        raise ValueError('Reference token budget admits no complete example')
    return result


def assign_guidance(rows, seed, guided_count, max_steps):
    """Assign before execution; do not inspect outcomes or model scores."""
    rng = random.Random(seed + 730001)
    chosen = set(rng.sample(sorted(r['rollout_id'] for r in rows),
                            min(guided_count, len(rows))))
    output = []
    for row in sorted(rows, key=lambda r: r['rollout_id']):
        row = dict(row, guidance_selected=row['rollout_id'] in chosen)
        if row['guidance_selected']:
            row['guidance_cut'] = rng.randint(1, max_steps // 2)
        output.append(row)
    return output


def request_condition(row):
    """Render the declared original goal, without consulting outcomes or plans.

    Uses the same six goal templates as the local HSL adapter. Official task
    metadata is a semantic resource; it conveys no object instance/location or
    next action. Sliced/nested requests are left unchanged because the current
    HSL type signature does not represent their additional constraints.
    """
    from alfworld_hindsight import goal_instruction
    params = json.loads((Path(row['gamefile']).parent/'traj_data.json').read_text())['pddl_params']
    if params.get('object_sliced') or params.get('mrecep_target'):
        return dict(status='unsupported_constraints', instruction=None, signature=None)
    destination = params['toggle_target'] if row['task_type']=='look_at_obj_in_light' else params['parent_target']
    signature = [row['task_type'], params['object_target'].lower(), destination.lower()]
    return dict(status='supported', signature=signature,
                instruction=goal_instruction(*signature), resource='official_original_task_metadata')


def bind_original_request(result, condition):
    """Bind only verified original-goal entries; keep every other HSL entry.

    Replaces the input condition in an existing slot. Masks, executed prefix,
    sampling weight and trajectory coefficient are unchanged; no duplicate
    demonstration or additional forward pass is introduced.
    """
    h = result['hindsight']
    h['original_task_description'] = result['task_description']
    matched = []
    if result['won'] and condition['status']=='supported':
        for goal in h['goals']:
            signature = (goal['task_type'], goal['object_type'], goal['destination_type'])
            if signature != tuple(condition['signature']):
                continue
            goal['canonical_instruction'] = goal['instruction']
            goal['instruction'] = result['task_description']
            goal['original_request_bound'] = True
            matched.append(goal['prefix_length'])
        if not matched:
            raise ValueError('Verified original goal missing from hindsight entries')
    h['original_request_binding'] = dict(verified=bool(result['won']),
        matched_prefixes=matched, semantic_condition=condition)
    return len(matched)


def guided_messages(messages, task, observation, bank):
    goal, obs = query_words(task), query_words(observation)
    def score(reference):
        a, b = query_words(reference['goal']), query_words(reference['context'])
        return (len(goal & a) / max(1, len(goal | a)) +
                .25 * len(obs & b) / max(1, len(obs | b)), reference['id'])
    reference = max(bank, key=score)
    guided = copy.deepcopy(messages)
    assert guided[0]['role'] in ('system','user')
    guided[0]['content'] += reference['hint']
    return guided, dict(reference_id=reference['id'], hint_tokens=reference['hint_tokens'])


def bounded_plan(env, timeout_seconds=5, action=None):
    """Solve once on the live state; hard deadline, zero extra environment steps.

    The native planner can block inside C. A fork inherits its current state,
    returns only actions, and leaves the collector's state untouched.
    """
    start = time.monotonic()
    parent, child = socket.socketpair()
    pid = os.fork()
    if pid == 0:
        parent.close()
        try:
            if action is not None:
                assert action in env.state.admissible_commands
                state, _, _ = env.step(action)
                won = bool(state.won)
                value = dict(status='complete', won=won, executed_action=action, expert_executed_actions=1,
                             feedback=str(state.feedback), plan=[] if won else env._pddl_state.replan(env._entity_infos))
            else:
                value = dict(status='complete', plan=env._pddl_state.replan(env._entity_infos))
        except Exception as exc:
            value = dict(status='error', error=f'{type(exc).__name__}: {exc}', plan=[])
        try:
            child.sendall(json.dumps(value).encode())
        finally:
            child.close()
            os._exit(0)
    child.close()
    data = bytearray()
    result = dict(status='timeout', plan=[])
    try:
        while True:
            remaining = timeout_seconds - (time.monotonic() - start)
            if remaining <= 0 or not select.select([parent], [], [], remaining)[0]:
                os.kill(pid, signal.SIGKILL)
                break
            chunk = parent.recv(65536)
            if not chunk:
                result = json.loads(data) if data else dict(status='error', plan=[])
                break
            data.extend(chunk)
    finally:
        parent.close()
        os.waitpid(pid, 0)
    result['seconds'] = time.monotonic() - start
    return result
