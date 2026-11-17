"""Native symbolic gate for a locally available original-goal operation.

Uses privileged task rules and current state, not a deployment action rule.
No action-name list: effects must lie in the goal's predicate dependency
closure and change a known target object, or directly satisfy a goal predicate
on a known destination. Only unconditional grounded positive effects are used.
"""
from alfworld_acquisition_correction import public_goal_candidates


def positive_predicates(condition, axioms=None, seen=None):
    axioms = axioms or {}
    seen = set() if seen is None else seen
    if hasattr(condition, 'predicate'):
        if getattr(condition, 'negated', False):
            return set()
        name = condition.predicate
        if name in axioms and name not in seen:
            return positive_predicates(axioms[name].condition, axioms, seen | {name})
        return {name}
    predicates = set()
    for part in getattr(condition, 'parts', ()):
        predicates.update(positive_predicates(part, axioms, seen))
    return predicates


def goal_predicate_context(env):
    native = env._pddl_state
    axioms = {axiom.name: axiom for axiom in native.task.axioms}
    goal = positive_predicates(native.task.goal, axioms)
    relevant = set(goal)
    while True:
        previous = set(relevant)
        for schema in native._actions.values():
            if any(not effect.literal.negated and effect.literal.predicate in relevant for effect in schema.effects):
                relevant.update(positive_predicates(schema.precondition, axioms))
        if relevant == previous:
            break
    return dict(goal_predicates=sorted(goal), relevant_predicates=sorted(relevant))


def available_goal_operations(env, state, condition, known, context):
    if not public_goal_candidates(condition, known):
        return {}
    _, object_type, destination_type = condition['signature']
    infos = env._entity_infos
    objects = {key for key, value in infos.items() if value.name in known
               and value.name.rsplit(' ', 1)[0] == object_type}
    destinations = {key for key, value in infos.items() if value.name in known
                    and value.name.rsplit(' ', 1)[0] == destination_type}
    facts = {(fact.name, tuple(argument.name for argument in fact.arguments)) for fact in env._pddl_state.facts}
    goal = set(context['goal_predicates'])
    relevant = set(context['relevant_predicates'])
    operations = {}
    for action, command in zip(state['_valid_actions'], state['_valid_commands']):
        mapping = {'?' + placeholder.name: variable.name for placeholder, variable in action.mapping.items()}
        evidence = []
        for effect in env._pddl_state._actions[action.name].effects:
            atom = effect.literal
            if atom.negated or atom.predicate not in relevant or effect.parameters:
                continue
            if effect.condition.__class__.__name__ != 'Truth':
                continue
            arguments = tuple(mapping.get(argument, argument) for argument in atom.args)
            if any(argument.startswith('?') for argument in arguments):
                continue
            if (atom.predicate, arguments) in facts:
                continue
            if not (objects.intersection(arguments) or (atom.predicate in goal and destinations.intersection(arguments))):
                continue
            evidence.append(dict(predicate=atom.predicate, arguments=list(arguments),
                public_arguments=[infos[arg].name if arg in infos else arg for arg in arguments]))
        if evidence:
            assert command in state.admissible_commands
            operations[command] = evidence
    return operations


def conjunctive_literals(condition):
    if hasattr(condition, 'predicate'):
        return [condition]
    if condition.__class__.__name__ == 'Truth':
        return []
    if condition.__class__.__name__ != 'Conjunction':
        raise ValueError('Goal adapter requires a conjunctive native condition: ' + str(condition))
    return [atom for part in condition.parts for atom in conjunctive_literals(part)]


def unify(pattern, value, mapping=None):
    if len(pattern) != len(value):
        return None
    mapping = dict(mapping or {})
    for argument, concrete in zip(pattern, value):
        if argument.startswith('?'):
            if argument in mapping and mapping[argument] != concrete:
                return None
            mapping[argument] = concrete
        elif argument != concrete:
            return None
    return mapping


def grounded_goal_context(env, sas_grounded=False):
    """Bind static goal constraints, retaining exact object/destination facts."""
    import itertools
    native = env._pddl_state
    goal = native.task.goal
    axioms = {axiom.name: axiom for axiom in native.task.axioms}
    if goal.predicate in axioms:
        axiom = axioms[goal.predicate]
        assert axiom.num_external_parameters == len(goal.args)
        mapping = {parameter.name: argument for parameter,argument in zip(axiom.parameters,goal.args)}
        atoms = conjunctive_literals(axiom.condition)
        parameters = {parameter.name: parameter.type_name for parameter in axiom.parameters}
    else:
        mapping, atoms, parameters = {}, [goal], {}
    dynamic = {effect.literal.predicate for schema in native._actions.values() for effect in schema.effects}
    facts = {(fact.name,tuple(v.name for v in fact.arguments)) for fact in native.facts}
    # A predicate can be dynamic for ordinary objects but immutable for a
    # particular grounded entity (e.g. a non-pickupable lamp's container).
    # Native unoptimized SAS variable domains enumerate its mutable ground
    # propositions; constants/unreachable propositions stay fixed in this task.
    mutable_ground = None
    if sas_grounded:
        import re
        mutable_ground = {(match.group(1),tuple(match.group(2).split(', ')) if match.group(2) else ())
            for match in re.finditer(r'^Atom ([^\(]+)\(([^\n]*)\)$',native.sas,re.MULTILINE)}
        assert mutable_ground, 'Native SAS atom interface must be verified.'
    bindings = [mapping]
    for atom in atoms:
        if atom.negated or atom.predicate in dynamic or atom.predicate == '=':
            continue
        updated = []
        for binding in bindings:
            for predicate,arguments in facts:
                if predicate == atom.predicate:
                    matched = unify(atom.args,arguments,binding)
                    if matched is not None:
                        updated.append(matched)
        bindings = updated
    expanded = []
    for binding in bindings:
        missing = [variable for variable in parameters if variable not in binding]
        domains = [[name for name,type_name in native.name2type.items() if type_name == parameters[variable]]
                   for variable in missing]
        for values in itertools.product(*domains):
            expanded.append(dict(binding,**dict(zip(missing,values))))
    clauses = []
    for binding in expanded:
        clause = []
        valid = True
        for atom in atoms:
            arguments = tuple(binding.get(argument,argument) for argument in atom.args)
            assert all(not argument.startswith('?') for argument in arguments)
            if atom.predicate == '=':
                satisfied = arguments[0] == arguments[1]
            else:
                satisfied = (atom.predicate,arguments) in facts
            is_dynamic = ((atom.predicate,arguments) in mutable_ground if sas_grounded else atom.predicate in dynamic)
            if not is_dynamic:
                if satisfied == bool(atom.negated):
                    valid = False
                    break
            else:
                assert not atom.negated, 'Negative dynamic goals require a separate verified adapter.'
                clause.append(dict(predicate=atom.predicate,arguments=list(arguments)))
        if valid:
            clauses.append(clause)
    assert clauses, 'No native grounded goal clause'
    return dict(clauses=clauses, mutability='native_SAS_ground_domains' if sas_grounded else 'predicate_class')


def available_grounded_goal_operations(env, state, condition, known, context):
    """Missing exact goal facts and object prerequisites of their producers.

    One dependency layer; candidate usefulness is subsequently checked by
    an actual native action branch, not inferred from this symbolic gate.
    """
    if not public_goal_candidates(condition,known):
        return {}
    infos = env._entity_infos
    objects = {key for key,value in infos.items() if value.name in known
               and value.name.rsplit(' ',1)[0] == condition['signature'][1]}
    facts = {(fact.name,tuple(v.name for v in fact.arguments)) for fact in env._pddl_state.facts}
    desired = {(atom['predicate'],tuple(atom['arguments'])) for clause in context['clauses'] for atom in clause
               if (atom['predicate'],tuple(atom['arguments'])) not in facts}
    patterns = set(desired)
    for predicate,arguments in desired:
        for schema in env._pddl_state._actions.values():
            for effect in schema.effects:
                atom = effect.literal
                if atom.negated or effect.parameters or atom.predicate != predicate:
                    continue
                binding = unify(atom.args,arguments)
                if binding is None:
                    continue
                for precondition in conjunctive_literals(schema.precondition):
                    if precondition.negated:
                        continue
                    grounded = tuple(binding.get(argument,argument) for argument in precondition.args)
                    if objects.intersection(grounded):
                        patterns.add((precondition.predicate,grounded))
    operations = {}
    for action,command in zip(state['_valid_actions'],state['_valid_commands']):
        binding = {'?' + placeholder.name: variable.name for placeholder,variable in action.mapping.items()}
        evidence = []
        for effect in env._pddl_state._actions[action.name].effects:
            atom = effect.literal
            if atom.negated or effect.parameters or effect.condition.__class__.__name__ != 'Truth':
                continue
            arguments = tuple(binding.get(argument,argument) for argument in atom.args)
            if any(argument.startswith('?') for argument in arguments) or (atom.predicate,arguments) in facts:
                continue
            matched = [list(pattern) for predicate,pattern in patterns if predicate == atom.predicate
                       and unify(pattern,arguments) is not None]
            if not matched:
                continue
            # The admitted operation itself only references public entities.
            from alfworld_acquisition_correction import plan_public_support
            if not plan_public_support([command],known)['supported']:
                continue
            evidence.append(dict(predicate=atom.predicate,arguments=list(arguments),matched_requirements=matched))
        if evidence:
            operations[command] = evidence
    return operations
