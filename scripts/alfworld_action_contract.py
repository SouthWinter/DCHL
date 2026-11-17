"""Explicit action-only dialogue contract for a separately trained baseline."""
import copy
import re

INSTRUCTION = '''Interact with a household to complete the stated task. Use the observations and your previous actions to select the next action.
Available actions: go to {recep}; take {obj} from {recep}; put {obj} in/on {recep}; open {recep}; close {recep}; toggle {obj} {recep}; clean {obj} with {recep}; heat {obj} with {recep}; cool {obj} with {recep}; examine {obj}; look; inventory.
An observation of "Nothing happened" means the action was invalid. Output exactly one line:
Action: <your next action>'''


def prefix():
    return [dict(role='user', content=INSTRUCTION), dict(role='assistant', content='OK')]


def canonical_messages(messages):
    result = copy.deepcopy(messages)
    result[0]['content'] = INSTRUCTION
    for message in result[2:]:
        if message['role'] == 'assistant':
            matches = re.findall(r'(?:^|\n)Action:\s*([^\n]+)', message['content'])
            if len(matches) != 1:
                raise ValueError('Expected exactly one action in demonstration reply')
            message['content'] = 'Action: ' + matches[0].strip()
            message.pop('loss_char_start', None)
    return result


def mask_repeated_navigation(steps, goal):
    """Mask inverse navigation pairs whose observations were already available.

    Preserve original inputs and all first observations. Restrict to exact
    inverse state deltas, avoiding assumptions about arbitrary replay deletion.
    """
    mask = list(goal['action_loss_mask'])
    reasons = list(goal['relevance_reasons'])
    seen = set()
    removed = set()
    prefix_steps = steps[:goal['prefix_length']]
    for i, step in enumerate(prefix_steps):
        if i + 1 < len(prefix_steps):
            other = prefix_steps[i + 1]
            both_go = all(s['action'].startswith('go to ') for s in (step, other))
            repeated = all((s['action'], s['result_observation']) in seen for s in (step, other))
            adds, deletes = set(map(tuple, step['added_facts'])), set(map(tuple, step['removed_facts']))
            inverse = bool(adds or deletes) and adds == set(map(tuple, other['removed_facts'])) and deletes == set(map(tuple, other['added_facts']))
            if both_go and repeated and inverse and i not in removed:
                removed.update((i, i + 1))
        seen.add((step['action'], step['result_observation']))
    for i in removed:
        mask[i] = False
        reasons[i] = 'repeated_observation_inverse_navigation'
    result = copy.deepcopy(goal)
    n, length = sum(mask), len(mask)
    if not n:
        raise ValueError('Navigation refinement removed all target support')
    result.update(action_loss_mask=mask, relevance_reasons=reasons,
                  relevant_actions=n, hsl_weight=(n / length) ** .8 * n,
                  relevance_rule='backward_with_repeated_inverse_navigation_v1')
    return result
