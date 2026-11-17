"""Stochastic guided policy rollouts for shared behavioral data valuation."""
import argparse
import json
import math
import os
import time
import urllib.request
from multiprocessing import get_context
from pathlib import Path

import collect_alfworld_takeover_frontier as core

CONFIG = {}


def init(config):
    global CONFIG
    CONFIG = config
    os.environ['ALFWORLD_DATA'] = config['data_root']
    if config.get('guidance_bank'):
        CONFIG['guidance_reference_bank'] = json.loads(Path(config['guidance_bank']).read_text())
        CONFIG['record_generation_usage'] = True


def collect(row):
    if os.environ.get('ALFWORLD_DIALOGUE_HSL')=='1':
        if CONFIG.get('correction_backend') == 'cycle_resample':
            from alfworld_cycle_resample import collect as cycle_collect
            return cycle_collect(row, CONFIG)
        if CONFIG.get('correction_backend') == 'budget_restart':
            from alfworld_budget_restart import collect as restart_collect
            return restart_collect(row, CONFIG)
        from alfworld_dialogue_hsl import collect as dialogue_collect
        return dialogue_collect(row,CONFIG)
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
    started = time.monotonic()
    result = dict(row, trajectory=[], status='running')
    policy_version = CONFIG.get('policy_version')
    if policy_version:
        result['policy_version'] = policy_version
    env = None
    try:
        if policy_version:
            with urllib.request.urlopen(CONFIG['endpoint'] + '/hsl/policy_version', timeout=30) as response:
                loaded = json.load(response)
            if loaded['policy_version'] != policy_version or loaded['pending_version'] is not None:
                raise ValueError('rollout service weights differ from collection policy version')
        env = textworld.start(row['gamefile'],
            request_infos=textworld.EnvInfos(won=True, admissible_commands=True),
            wrappers=[AlfredDemangler(shuffle=False)])
        state = env.reset()
        task = core.task_description(str(state.feedback))
        history = []
        for step in range(CONFIG['max_steps']):
            observation = str(state.feedback)
            actions = [str(a) for a in state.admissible_commands if a != 'help']
            messages = [{'role': 'system', 'content': core.SYSTEM_PROMPT},
                        {'role': 'user', 'content': core.prompt(task, observation, actions, history)}]
            payload = dict(model='qwen3-4b', messages=messages,
                temperature=1.0, top_p=1.0, top_k=-1, max_tokens=64,
                presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
                seed=CONFIG['seed'] + row['rollout_id'] * 1009 + step,
                logprobs=CONFIG.get('collect_logprobs', True), top_logprobs=0,
                chat_template_kwargs={'enable_thinking': False},
                guided_choice=[f'<action>{a}</action>' for a in actions])
            if not CONFIG.get('collect_logprobs', True):
                payload.pop('top_logprobs')
            request = urllib.request.Request(CONFIG['endpoint'] + '/v1/chat/completions',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=180) as response:
                output = json.load(response)
            choice = output['choices'][0]
            content = choice['message']['content']
            match = core.ACTION_RE.search(content)
            action = match.group(1).strip() if match else ''
            if action not in actions or choice['finish_reason'] != 'stop':
                raise ValueError(f'Incomplete/non-admissible response: {content!r}')
            tokens = choice['logprobs']['content'] if CONFIG.get('collect_logprobs', True) else []
            if CONFIG.get('collect_logprobs', True) and (not tokens or not all(math.isfinite(t['logprob']) for t in tokens)):
                raise ValueError('Missing or non-finite sampling log probabilities')
            state, _, done = env.step(action)
            result['trajectory'].append(dict(step=step, messages=messages,
                observation=observation, admissible_actions=actions,
                action=action, response=content, token_logprobs=tokens,
                response_logprob=sum(t['logprob'] for t in tokens) if tokens else None,
                result_observation=str(state.feedback), won=bool(state.won), done=bool(done)))
            history.append(dict(observation=observation, action=action))
            if done or state.won:
                break
        result.update(status='complete', won=bool(state.won), reward=float(state.won),
                      steps=len(result['trajectory']), task_description=task)
    except Exception as exc:
        result.update(status='infrastructure_error', error=f'{type(exc).__name__}: {exc}')
    finally:
        if env is not None:
            env.close()
    if result['status'] == 'complete' and CONFIG.get('hindsight', False):
        try:
            from alfworld_hindsight import replay_and_relabel
            result['hindsight'] = replay_and_relabel(result, CONFIG.get('relevance_rule', 'backward'))
            result['hindsight']['engineering_fixture_only'] = False
        except Exception as exc:
            result.update(status='relabel_error', error=f'{type(exc).__name__}: {exc}')
    result['wall_seconds'] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20260822)
    parser.add_argument('--policy-version', help='Require the HSL server to acknowledge this loaded version')
    parser.add_argument('--no-logprobs', dest='collect_logprobs', action='store_false')
    parser.add_argument('--hindsight', action='store_true', help='Relabel inside each CPU collection worker')
    parser.add_argument('--relevance-rule', choices=['backward', 'replay_search'], default='backward')
    parser.add_argument('--guidance-bank', type=Path)
    parser.add_argument('--guided-count', type=int, default=4)
    parser.add_argument('--correction-backend', choices=['bounded_planner','equivalent_condition','public_grounded_planner','public_acquisition_planner','cost_screened_label','goal_operation_label','budget_restart','cycle_resample'])
    parser.add_argument('--planner-timeout', type=float, default=5)
    parser.add_argument('--record-generation-usage', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--skip-hindsight-relabel', action='store_true')
    args = parser.parse_args()
    if args.guidance_bank and args.correction_backend:
        parser.error('Choose reference guidance or environment-expert correction')
    # Online HSL launches a fresh collector each round. Read an explicit
    # run-local throughput override here so active trainers need no restart.
    if args.hindsight and args.output.parent.parent.name == 'rounds':
        control = args.output.parent.parent.parent.parent / 'throughput.json'
        if control.exists():
            args.workers = int(json.loads(control.read_text())['rollout_workers'])
            if args.workers < 1:
                parser.error('rollout_workers must be positive')
    if args.hindsight and not args.policy_version:
        parser.error('--hindsight requires --policy-version')
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    completed = set()
    if args.output.exists():
        completed = {r['rollout_id'] for r in map(json.loads, args.output.read_text().splitlines())
                     if r['status'] == 'complete'}
    planned = [dict(row, task_index=i, repeat=j, rollout_id=i*args.repeats+j)
               for i, row in enumerate(tasks) for j in range(args.repeats)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.guidance_bank or (args.correction_backend and args.correction_backend not in ('budget_restart','cycle_resample')):
        from alfworld_inline_correction import assign_guidance
        planned = assign_guidance(planned,args.seed,args.guided_count,args.max_steps)
        if args.correction_backend in ('equivalent_condition','goal_operation_label'):
            for row in planned:
                if row['guidance_selected']:
                    row['guidance_cut'] = 0
        (args.output.parent/'guidance_assignment.json').write_text(json.dumps([
            {k:r[k] for k in ('rollout_id','guidance_selected','guidance_cut') if k in r}
            for r in planned],indent=2)+'\n')
    work = [r for r in planned if r['rollout_id'] % args.num_shards == args.shard_index
            and r['rollout_id'] not in completed]
    (args.output.parent / (args.output.stem + '_execution.json')).write_text(
        json.dumps(dict(workers=args.workers, pending_rollouts=len(work),
                        seed=args.seed, policy_version=args.policy_version), indent=2) + '\n')
    errors = 0
    with args.output.open('a') as writer, get_context('spawn').Pool(
            args.workers, initializer=init, initargs=(vars(args),), maxtasksperchild=8) as pool:
        for i, result in enumerate(pool.imap_unordered(collect, work), 1):
            writer.write(json.dumps(result, ensure_ascii=False)+'\n')
            writer.flush()
            errors += result['status'] != 'complete'
            print(json.dumps(dict(completed=i, total=len(work), errors=errors,
                                  rollout_id=result['rollout_id'], status=result['status'])), flush=True)
    if errors:
        raise SystemExit(f'{errors} failed rollouts; successful rows retained for resume')


if __name__ == '__main__':
    main()
