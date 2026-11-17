"""Let students acquire public information before bounded expert completion.

The ALFWorld adapter checks concrete plan entities. This is a conservative
support test, not a certificate of optimal partial-observation planning.
"""
from audit_alfworld_expert_observability import ENTITY


def observe_entities(observation, known):
    known.update(ENTITY.findall(observation.lower()))


def public_goal_candidates(condition, known):
    if condition['status'] != 'supported':
        return False
    family, object_type, destination_type = condition['signature']
    need = 2 if family == 'pick_two_obj_and_place' else 1
    objects = {e for e in known if e.rsplit(' ', 1)[0] == object_type}
    destinations = {e for e in known if e.rsplit(' ', 1)[0] == destination_type}
    return len(objects) >= need and bool(destinations)


def plan_public_support(plan, known):
    dependencies = {e for command in plan for e in ENTITY.findall(command.lower())}
    unknown = dependencies - known
    return dict(supported=not unknown, referenced_entities=sorted(dependencies),
                unknown_entities=sorted(unknown), known_entities=sorted(known),
                scope='Literal concrete entities in the complete plan were observed before handoff; locations, intent and full belief sufficiency remain separate')
