"""Matched post-update rollouts; feedback trajectories never enter the buffer."""
import json
import os
from pathlib import Path
import subprocess
import time


def paired_outcomes(before, after):
    def index(rows):
        result = {r['rollout_id']: r for r in rows if r['status'] == 'complete'}
        if len(result) != len(rows):
            raise ValueError('Pairing requires unique completed rollouts')
        return result
    left, right = index(before), index(after)
    if not left or left.keys() != right.keys():
        raise ValueError('Before/after task IDs differ')
    pairs = []
    for key, a in left.items():
        b = right[key]
        if a['gamefile'] != b['gamefile'] or a['task_index'] != b['task_index']:
            raise ValueError('Before/after task identity differs')
        # Validate actual reset observations and legal actions, not just file IDs.
        x, y = a['trajectory'][0], b['trajectory'][0]
        if x['observation'] != y['observation'] or x['admissible_actions'] != y['admissible_actions']:
            raise ValueError('Before/after initial environment states differ')
        pairs.append(dict(rollout_id=key, gamefile=a['gamefile'],
                          before=int(a['won']), after=int(b['won']),
                          change=int(b['won']) - int(a['won'])))
    return dict(pairs=pairs, rollout_count=len(pairs),
                before_successes=sum(p['before'] for p in pairs),
                after_successes=sum(p['after'] for p in pairs),
                improved=sum(p['change'] == 1 for p in pairs),
                regressed=sum(p['change'] == -1 for p in pairs))


def collect_post_update(config, step, version, root):
    directory = root / 'rounds' / f'{step:05d}'
    output = directory / 'paired_post.jsonl'
    workers = int(config.hsl.get('workers', 6))
    control = root.parent / 'throughput.json'
    if control.exists():
        workers = int(json.loads(control.read_text())['rollout_workers'])
    command = [str(config.hsl.environment_python), '-u',
        str(Path(__file__).with_name('collect_alfworld_value_calibration.py')),
        '--tasks', str(directory / 'tasks.jsonl'), '--output', str(output),
        '--endpoint', config.hsl.endpoint, '--data-root', config.hsl.data_root,
        '--policy-version', version, '--no-logprobs',
        '--num-shards', '1', '--shard-index', '0', '--repeats', '1',
        '--workers', str(workers), '--max-steps', '40',
        '--seed', str(int(config.trainer.seed) + step * 100003)]
    started = time.monotonic()
    with (directory / 'paired_collector.log').open('a') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, OMP_NUM_THREADS='1'), check=True)
    def completed(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return list({r['rollout_id']: r for r in rows if r['status'] == 'complete'}.values())
    before, after = completed(directory / 'rollouts.jsonl'), completed(output)
    if len(after) != int(config.hsl.get('rollouts_per_step', 18)):
        raise ValueError('Missing post-update feedback rollouts')
    if any(r['policy_version'] != version for r in after):
        raise ValueError('Stale post-update policy version')
    result = paired_outcomes(before, after)
    result.update(collection_seconds=time.monotonic() - started,
                  policy_version=version, decoding_seed=int(config.trainer.seed) + step * 100003)
    (directory / 'paired_feedback.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
