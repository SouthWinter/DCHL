"""Goal certificates from public action/response pairs, without state replay.

The ALFWorld adapter recognizes successful tool responses. Certificates own
both the supervision support and the feedback positions. No environment facts,
teacher predictions, expert plans, or old hindsight labels enter this module.
"""
import re
import time

PLACE = 'pick_and_place_simple'
TWO = 'pick_two_obj_and_place'
LIGHT = 'look_at_obj_in_light'
PROPERTIES = {'heat': 'pick_heat_then_place_in_recep',
              'cool': 'pick_cool_then_place_in_recep',
              'clean': 'pick_clean_then_place_in_recep'}

def kind(entity):
    return re.sub(r' \d+$', '', entity)

def execution_event(step):
    """Accept only a command paired with its exact public success receipt."""
    action = step['action']; response = step['result_observation'].strip()
    patterns = [('take', r'take (.+) from (.+)', 'You pick up the {} from the {}.'),
                ('move', r'move (.+) to (.+)', 'You move the {} to the {}.'),
                ('heat', r'heat (.+) with (.+)', 'You heat the {} using the {}.'),
                ('cool', r'cool (.+) with (.+)', 'You cool the {} using the {}.'),
                ('clean', r'clean (.+) with (.+)', 'You clean the {} using the {}.')]
    for verb, pattern, receipt in patterns:
        match = re.fullmatch(pattern, action)
        if match and response == receipt.format(*match.groups()):
            return (verb, *match.groups())
    if action.startswith('use '):
        obj = action[4:]
        if response == f'You turn on the {obj}.': return ('on', obj)
        if response == f'You turn off the {obj}.': return ('off', obj)
    if action.startswith('go to '):
        loc = action[6:]
        if response.startswith(f'You arrive at {loc}.'): return ('go', loc)
    for verb in ('open', 'close'):
        if action.startswith(verb+' '):
            obj = action[len(verb)+1:]
            if response.startswith(f'You {verb} the {obj}.'): return (verb, obj)
    return None

def relabel(row):
    started = time.monotonic()
    # Explicit allowlist makes privileged trajectory fields inaccessible below.
    steps = [{k: s[k] for k in ('action', 'observation', 'result_observation')}
             for s in row['trajectory']]
    held = None; location = None; navigation = None
    support = {}; properties = {}; placed = {}; lamps = {}; opened = {}
    search = set(); goals = []; seen = set(); events = []

    def emit(task, obj_type, destination, instruction, required, decisive, witnesses):
        key = (task, obj_type, destination)
        if key in seen: return
        seen.add(key)
        # Preserve successful exploration; state-changing supervision is drawn
        # only from the certificate's dependency support. All history is input.
        selected = set(required) | search
        mask = [j in selected for j in range(i+1)]
        n = sum(mask)
        assert n and decisive and set(decisive) <= selected
        goals.append(dict(task_type=task, object_type=obj_type,
            destination_type=destination, instruction=instruction,
            first_achievement_step=i+1, prefix_length=i+1,
            action_loss_mask=mask, relevant_actions=n,
            hsl_weight=(n/(i+1))**.8*n,
            relevance_reasons=['certificate_support_or_search' if x else 'outside_certificate' for x in mask],
            relevance_rule='public_execution_certificate_v1',
            feedback_steps=sorted(set(decisive)),
            execution_certificate=dict(witnesses=witnesses, support_steps=sorted(required),
                                       feedback_steps=sorted(set(decisive)))))

    for i, step in enumerate(steps):
        event = execution_event(step)
        if event is None: continue
        events.append(dict(step=i, event=list(event)))
        verb, *args = event
        if verb == 'go':
            location=args[0]; navigation=i; search.add(i)
        elif verb == 'open':
            opened[args[0]]=i; search.add(i)
        elif verb == 'close':
            opened.pop(args[0], None)
        elif verb == 'take':
            obj, source=args; held=obj
            previous=placed.pop(obj, None)
            support[obj]=set(previous['support'] if previous else ()) | {i}
            if navigation is not None: support[obj].add(navigation)
            if source in opened: support[obj].add(opened[source])
        elif verb in PROPERTIES:
            obj, tool=args
            # A success receipt validates the transformation; an observed take
            # supplies the input provenance needed for a training certificate.
            if held != obj or obj not in support: continue
            chain=set(support[obj]) | {i}
            if navigation is not None: chain.add(navigation)
            properties[(obj, verb)]=(i, chain)
            if verb in ('heat','cool'):
                properties.pop((obj, 'cool' if verb=='heat' else 'heat'), None)
        elif verb == 'move':
            obj, dest=args
            if held != obj or obj not in support: continue
            chain=set(support[obj]) | {i}
            if navigation is not None: chain.add(navigation)
            if dest in opened: chain.add(opened[dest])
            held=None
            placed[obj]=dict(destination=dest, support=chain, step=i)
            ot, dt=kind(obj),kind(dest)
            emit(PLACE,ot,dt,f'put a {ot} in/on the {dt}.',chain,[i],[dict(object=obj,destination=dest)])
            for operation,task in PROPERTIES.items():
                evidence=properties.get((obj,operation))
                if evidence:
                    position,prop_chain=evidence
                    emit(task,ot,dt,f'{operation} a {ot} and put it in/on the {dt}.',
                         chain|prop_chain,[position,i],[dict(object=obj,destination=dest,operation=operation)])
            peers=[(o,p) for o,p in placed.items() if o!=obj and kind(o)==ot and p['destination']==dest]
            if peers:
                other,other_proof=peers[0]
                emit(TWO,ot,dt,f'put two {ot} in/on the {dt}.',chain|other_proof['support'],
                     [other_proof['step'],i],[dict(object=o,destination=dest) for o in (other,obj)])
        elif verb == 'on':
            lamp=args[0]
            if kind(lamp) in ('desklamp','floorlamp') and location is not None:
                lamps[lamp]=(location,i)
        elif verb == 'off': lamps.pop(args[0],None)
        if held is not None and held in support:
            for lamp,(loc,position) in lamps.items():
                if loc==location:
                    chain=set(support[held])|{position}
                    if navigation is not None: chain.add(navigation)
                    emit(LIGHT,kind(held),kind(lamp),f'examine the {kind(held)} with the {kind(lamp)}.',
                         chain,[position],[dict(object=held,lamp=lamp,location=loc)])
    return dict(gamefile=row['gamefile'], requested_task_type=row['task_type'],
        rollout_id=row.get('rollout_id'), policy_version=row['policy_version'],
        source_success=bool(row['won']), steps=steps, goals=goals,
        relabeler='public_execution_certificate_v1', privileged_state=False,
        action_relevance_ready=True, training_ready=False,
        relevance_mode='public_execution_certificate_v1', relevance_worker_seconds=0.,
        execution_events=events, dialogue_hsl=True, response_contract='action_only_v1',
        wall_seconds=time.monotonic()-started)
