"""Observed operation structure for goal-conditioned learner feedback."""
import re


def structure(actions):
    slots={}
    def replace(match):
        token=match.group()
        if token not in slots:slots[token]=f'[I{len(slots)}]'
        return slots[token]
    return tuple(re.sub(r'\[I\d+\]',replace,a) for a in actions)
