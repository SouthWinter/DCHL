"""BabyAI native goals, visible macro actions and replay-verified supervision."""
import copy
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('BABYAI_DATA_DIR', ROOT / 'data/babyai_v1'))
sys.path.append(str(ROOT / 'external/babyai_runtime/lib/python3.12/site-packages'))
from screen_babyai import BabyAI, all_levels, candidates, encode, decode, V
from textcraft_adapter import context_messages, action_records, training_records
import gymnasium
import minigrid

# Gymnasium discovers plugin entry points only on its first import. The shared
# TextCraft evaluator/trainer can import it before BabyAI's runtime is visible.
# Register explicitly in that import order as well as in standalone preparation.
if 'BabyAI-GoToRedBallGrey-v0' not in gymnasium.registry:
    minigrid.register_minigrid_envs()
assert all(name in gymnasium.registry for name in all_levels.values())

MAX_STEPS = 20
CONTEXT = 8192
SYSTEM = ('You control an agent in a room. Follow the goal, including its ordering constraints. '
          'Reply with exactly one action from the available actions, without explanation.')


def parse_action(text):
    value = re.split(r'Action\s*:\s*', text, flags=re.I)[-1].strip()
    try:
        decoded = json.loads(value)
        if isinstance(decoded, str):
            value = decoded
        elif isinstance(decoded, list) and len(decoded) == 1 and isinstance(decoded[0], str):
            value = decoded[0]
    except ValueError:
        pass
    return value.split('\n')[0].strip(' `"\'').lower() if value else ''


class NativeBabyAI(BabyAI):
    """Preserve native mission order and stop macro execution on native terminal."""
    def reset(self):
        super().reset()
        self.goal = self.env.unwrapped.mission
        self.native_terminated = self.native_truncated = False
        self.observer = None
        self.last_primitive_actions = []
        self.object_ids = {id(self.env.unwrapped.grid.get(x, y)): [x, y]
            for x in range(self.env.unwrapped.grid.width)
            for y in range(self.env.unwrapped.grid.height)
            if self.env.unwrapped.grid.get(x, y) is not None}

    def identity(self, obj):
        return None if obj is None else [obj.type, obj.color, self.object_ids.get(id(obj))]

    def intent(self, action):
        """Ground the intended target before execution, even for early success."""
        if action not in self.action_space:
            return dict(available=False, target=None)
        native = self.env.unwrapped
        pos = list(native.agent_pos)
        direction = int(native.agent_dir)
        delta = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        interactions = []
        for primitive in self.action_space[action]:
            if primitive == 0:
                direction = (direction-1) % 4
            elif primitive == 1:
                direction = (direction+1) % 4
            elif primitive == 2:
                pos = [pos[i]+delta[direction][i] for i in range(2)]
            elif primitive in (3, 4, 5):
                front = [pos[i]+delta[direction][i] for i in range(2)]
                interactions.append(self.identity(native.carrying if primitive == 4 else native.grid.get(*front)))
        if action.startswith('go to '):
            front = [pos[i]+delta[direction][i] for i in range(2)]
            target = self.identity(native.grid.get(*front))
        elif action.startswith('go through '):
            target = self.identity(native.grid.get(*pos))
        elif action.startswith(('pick', 'toggle', 'drop')):
            target = interactions
        else:
            target = None
        return dict(available=True, target=target)

    def step(self, action):
        self.last_primitive_actions = []
        valid, error = self.check_action_is_valid(action)
        self.infos['action_is_valid'] = valid
        if not valid:
            self.update_info(action, error)
            return self._get_obs(), self.reward, self.done, self.infos
        if 'check' in action:
            self.update_info(action, 'You can take the following actions: ' + ', '.join(self._get_action_space()))
            return self._get_obs(), self.reward, self.done, self.infos
        primitives = self.action_space[action]
        if not primitives:
            self.update_info(action, 'No change in state.')
            return self._get_obs(), self.reward, self.done, self.infos
        for primitive in primitives:
            obs, reward, terminated, truncated, infos = self.env.step(primitive)
            self.last_primitive_actions.append(int(primitive))
            if self.observer is not None:
                self.observer(primitive)
            changed = self.verify_action(primitive, obs)
            self.obs_2d = obs['image']
            if terminated or truncated or not changed:
                break
        self.native_terminated, self.native_truncated = terminated, truncated
        self.reward = float(reward)
        self.done = bool(terminated or truncated)
        description, actions = self.postprocess_obs(obs)
        if terminated and reward > 0:
            description += '\nThe task is completed.'
        elif self.done:
            description += '\nThe episode ended without completing the task.'
        self.states.append(description)
        self.action_space = actions
        self.steps += 1
        self.infos.update(infos)
        self.infos.update(state=description, steps=self.steps, action_is_valid=True)
        return description, reward, self.done, self.infos


class BabySession:
    def __init__(self):
        self.env = None

    def reset(self, example, goal=None, watch=False):
        idx = int(example['id'].split('_')[-1])
        level = all_levels[idx % 40+1]
        if self.env is None or self.env.game_name != level:
            self.close()
            self.env = NativeBabyAI(game_name=level, seed=idx//40)
        else:
            self.env.seed = idx//40
            self.env.reset()
        native = self.env.env.unwrapped
        if goal is not None:
            native.instrs = decode(goal)
            self.env.goal = native.instrs.surface(native)
            native.instrs.reset_verifier(native)
        self.example = example
        self.goal_spec = encode(native.instrs)
        self.description = self.env.goal
        self.messages = [dict(role='system', content=SYSTEM),
            dict(role='user', content='Your goal: '+self.description+'\n'+self.observation())]
        self.trajectory = []
        self.won = self.done = False
        self.score = 0.
        self.hits = {}
        self.goals = candidates(self.env) if watch else []
        self.validators = []
        for g in self.goals:
            verifier = decode(g['spec'])
            verifier.reset_verifier(native)
            initially_done = isinstance(verifier, V.GoToInstr) and verifier.verify(native.actions.done) == 'success'
            self.validators.append(None if initially_done else verifier)
        if watch:
            def observer(action):
                for i, verifier in enumerate(self.validators):
                    if verifier is None or i in self.hits:
                        continue
                    if int(action) == int(native.actions.drop):
                        native.update_objs_poss(verifier)
                    if verifier.verify(action) == 'success':
                        self.hits[i] = len(self.trajectory)
            self.env.observer = observer
        return self.messages[-1]['content']

    def observation(self):
        return self.env._get_obs()+'\nAvailable actions: '+json.dumps(list(self.env.action_space))

    def step(self, response, *, normalized=False):
        if self.done:
            raise RuntimeError('step after native episode termination')
        # A replay receives the exact action already sent to the environment.
        # Parsing it again can turn an invalid generated action into a valid
        # one (e.g. a trailing carriage return), changing the recorded state.
        action = response if normalized else parse_action(response)
        target = self.env.intent(action)
        self.env.step(action)
        self.won = bool(self.env.native_terminated and self.env.reward > 0)
        self.done = self.env.done
        self.score = 100.*self.won
        observation = self.observation()
        self.trajectory.append(dict(step=len(self.trajectory)+1, action=action, response=response,
            observation=observation, valid=target['available'], won=self.won, score=self.score,
            intent=target, primitive_actions=list(self.env.last_primitive_actions)))
        self.messages.extend([dict(role='assistant', content=action), dict(role='user', content=observation)])
        return observation, self.won

    def replay(self, example, actions, goal=None, watch=False):
        self.reset(example, goal, watch)
        for action in actions:
            self.step(action, normalized=True)
            if self.done:
                break
        return dict(won=self.won, score=self.score, description=self.description,
            trajectory=copy.deepcopy(self.trajectory), messages=copy.deepcopy(self.messages))

    def close(self):
        if self.env is not None:
            self.env.env.close()
            self.env = None


def structural_family(spec):
    # Native task structure and object roles; color and instance IDs do not
    # fragment feedback across otherwise equivalent TRAIN tasks.
    result = {}
    for key, value in spec.items():
        if key in ('desc', 'desc_move', 'desc_fixed'):
            result[key] = dict(type=value['type'], loc=value['loc'])
        elif key in ('instr_a', 'instr_b'):
            result[key] = structural_family(value)
        else:
            result[key] = value
    return result


def _relabel_impl(example, trajectory, tokenizer, version, session, targets=None):
    actions = [t['action'] for t in trajectory]
    original = session.replay(example, actions, watch=True)
    assert [(t['action'], t['observation']) for t in original['trajectory']] == [(t['action'], t['observation']) for t in trajectory], ('source_replay', example['id'])
    specs = copy.deepcopy(session.goals)
    hits = dict(session.hits)
    original_kind = session.goal_spec['kind']
    goals = []
    checks = 1
    for idx, end in hits.items():
        goal = specs[idx]
        full = session.replay(example, actions[:end+1], goal['spec'])
        checks += 1
        if not full['won']:
            continue
        length = len(full['trajectory'])
        # Only the final observation may change when a relabeled macro ends early.
        assert all(a['observation'] == b['observation'] and a['intent'] == b['intent']
                   for a, b in zip(trajectory[:length-1], full['trajectory'][:length-1])), ('goal_prefix_replay', example['id'], goal['instruction'])
        keep = list(range(length))
        for j in reversed(range(length)):
            trial_indices = [k for k in keep if k != j]
            trial = session.replay(example, [actions[k] for k in trial_indices], goal['spec'])
            checks += 1
            if (trial['won'] and len(trial['trajectory']) == len(trial_indices)
                and all(t['intent'] == trajectory[k]['intent'] for k, t in zip(trial_indices, trial['trajectory']))):
                keep = trial_indices
        assert keep, ('empty_support', example['id'], goal['instruction'])
        records = action_records(full['messages'][:-1], tokenizer)
        n = len(keep)
        goals.append(dict(instruction=full['description'], task_type=goal['spec']['kind'],
            object_type=json.dumps(structural_family(goal['spec']), sort_keys=True), native_goal=goal['spec'],
            prefix_length=length, relevant_actions=n, action_loss_mask=[j in keep for j in range(length)],
            feedback_steps=keep, hsl_weight=(n/length)**.8*n, records=records))
    return dict(gamefile=example['id'], policy_version=version, action_relevance_ready=True,
        requested_task_type=original_kind, source_success=original['won'], goals=goals,
        candidate_count=len(hits), relevance_checks=checks,
        label_source='native complete task predicates; same TRAIN world; physical-target-preserving reverse deletion')


def relabel(example, trajectory, tokenizer, version, session, targets=None):
    try:
        return _relabel_impl(example, trajectory, tokenizer, version, session, targets)
    except Exception:
        import traceback
        import time
        path = ROOT/'analysis/babyai_relabel_errors'
        path.mkdir(exist_ok=True)
        (path/f"{example['id']}_{time.time_ns()}.json").write_text(json.dumps(dict(
            example=example, trajectory=trajectory, version=version, traceback=traceback.format_exc()), ensure_ascii=False, indent=2))
        raise
