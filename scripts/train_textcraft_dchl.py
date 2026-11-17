"""Unchanged DCHL allocation plus native-verified TextCraft continuations."""
import copy
import json
import random
import re
import time
import train_textcraft_hsl as base
from train_alfworld_direct_candidate import DirectCandidateTrainer
from alfworld_decision_coverage import DecisionCoverageBuffer
from alfworld_goal_value_loop import BufferValueController
import alfworld_verified_recovery as recovery
import alfworld_batch_selection as selection
from textcraft_adapter import DATA, TextSession, action_records, training_records


def words(text):
    return set(re.findall(r'[a-z]+', text.lower()))


def compress_successful_suffix(session, example, original, cut):
    actions = [t['action'] for t in original['trajectory']]
    best, checks = original, 0
    for index in reversed(range(cut, len(actions))):
        if checks >= 24:
            break
        trial = session.replay(example, actions[:index]+actions[index+1:])
        checks += 1
        if trial['won']:
            best = trial
            actions = [t['action'] for t in trial['trajectory']]
    assert best['won'] and len(best['trajectory']) > cut
    return best, checks


class TextCraftDCHLTrainer(base.TextCraftTrainer, DirectCandidateTrainer):
    def collect_round(self, step, tasks, rng, version, root):
        labels, metrics = super().collect_round(step, tasks, rng, version, root)
        directory = root/'rounds'/f'{step:05d}'
        rows = [json.loads(line) for line in (directory/'rollouts.jsonl').read_text().splitlines()]
        examples = {e['id']: e for e in tasks}
        if not hasattr(self, 'reference_bank'):
            self.reference_bank = []
            for path in sorted((DATA/'replayed').glob('*.json')):
                row = json.loads(path.read_text())
                if row['won'] and row['id'] in examples:
                    self.reference_bank.append(dict(id=row['id'], goal=row['description'],
                        actions=[t['action'] for t in row['trajectory']]))
        failed = [row for row in rows if not row['won'] and len(row['trajectory']) > 1]
        picked = random.Random(int(self.config.trainer.seed)+step*8191).sample(failed, min(4, len(failed)))
        jobs = []
        for row in picked:
            cut = min(24, max(1, len(row['trajectory'])//2))
            for index, action in enumerate(row['trajectory'][:cut+1]):
                if index > 0 and not action['valid']:
                    cut = index
                    break
            target = words(row['description'])
            eligible = [entry for entry in self.reference_bank if entry['id'] != row['id']]
            reference = max(eligible, key=lambda entry:
                (len(words(entry['goal']) & target)/max(1, len(words(entry['goal']) | target)), entry['id']))
            jobs.extend(dict(row=row, cut=cut, reference=reference, attempt=attempt) for attempt in range(2))
        started = time.monotonic()
        tokenizer = self.model_config.tokenizer

        def execute(job):
            if not hasattr(self.textcraft_local, 'session'):
                self.textcraft_local.session = TextSession()
                self.textcraft_sessions.append(self.textcraft_local.session)
            session = self.textcraft_local.session
            row, cut = job['row'], job['cut']
            example = examples[row['id']]
            replay = session.replay(example, [t['action'] for t in row['trajectory'][:cut]])
            assert not replay['won']
            assert [(t['action'], t['observation']) for t in replay['trajectory']] == [
                (t['action'], t['observation']) for t in row['trajectory'][:cut]]
            route = (self.try_native_route(session, example, row, cut, job)
                     if hasattr(self, 'try_native_route') else None)
            hint = ('\nA successful example from another TRAIN task follows. Adapt it to the current task and state.\n'
                +job['reference']['goal']+'\n'+'\n'.join(job['reference']['actions']))
            generated = 0
            for turn in range(0 if route and route['result'] else 24):
                messages = copy.deepcopy(session.messages)
                messages[0]['content'] += hint
                answer = base.request(self.config.hsl.endpoint, messages, tokenizer,
                    int(self.config.trainer.seed)+step*100003+row['rollout_id']*1009+job['attempt']*503+turn)
                session.step(answer)
                generated += 1
                if session.done:
                    break
            result = dict(status='complete', rollout_id=row['rollout_id'], source_id=row['id'], cut=cut,
                attempt=job['attempt'], reference_id=job['reference']['id'], won=session.won,
                generated_steps=generated, trajectory=copy.deepcopy(session.trajectory), compression_checks=0)
            if route is not None:
                result['train_route'] = {k:v for k,v in route.items() if k != 'result'}
                result['train_route']['accepted'] = bool(route['result'])
                if route['result']:
                    result['original_reference_id'] = result['reference_id']
                    result['reference_id'] = row['id']
                    result['reference_kind'] = 'same_task_TRAIN_native_verified_suffix'
            if session.won:
                best = session.replay(example, [t['action'] for t in session.trajectory])
                assert best['won']
                if hasattr(self, 'make_recovery_goal'):
                    goal, support = self.make_recovery_goal(session, example, best, cut, tokenizer, row['description'])
                    result['compression_checks'] = support['checks']
                    result['history_support'] = support
                else:
                    best, checks = compress_successful_suffix(session, example, best, cut)
                    result['compression_checks'] = checks
                    records = action_records(best['messages'][:-1], tokenizer)
                    mask = [i >= cut for i in range(len(records))]
                    assert any(mask)
                    goal = dict(instruction=row['description'], task_type='craft', prefix_length=len(records),
                                action_loss_mask=mask, records=records)
                result['entry'] = dict(key=f"{step}:{row['rollout_id']}:{cut}", round=step, cut=cut,
                    rollout=dict(gamefile=row['id'], rollout_id=row['rollout_id'], policy_version=version),
                    goal=goal)
            return result

        results = list(self.textcraft_pool.map(execute, jobs))
        (directory/'recovery_jobs.jsonl').write_text(''.join(json.dumps(dict(rollout_id=j['row']['rollout_id'],
            cut=j['cut'], attempt=j['attempt'], reference_id=j['reference']['id']))+'\n' for j in jobs))
        (directory/'recoveries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
        for label, row in zip(labels, rows):
            label['_verified_recoveries'] = [r['entry'] for r in results
                if r['rollout_id'] == row['rollout_id'] and r.get('entry')]
        metrics.update(recovery_seconds=time.monotonic()-started, recovery_attempts=len(results),
            recovery_successes=sum(r['won'] for r in results),
            recovery_generated_steps=sum(r['generated_steps'] for r in results))
        route_trials = [r['train_route'] for r in results if 'train_route' in r]
        history_trials = [r['history_support'] for r in results if 'history_support' in r]
        if history_trials:
            metrics.update(recovery_history_successes=len(history_trials),
                recovery_history_validity_fallbacks=sum(r['original_validity_fallback'] for r in history_trials),
                recovery_history_supervised_actions=sum(len(r['selected']) for r in history_trials))
        if route_trials:
            metrics.update(train_route_trials=len(route_trials),
                train_route_successes=sum(r['accepted'] for r in route_trials),
                train_route_native_checks=sum(r['checks'] for r in route_trials),
                train_route_executed_steps=sum(r['executed_steps'] for r in route_trials),
                train_route_worker_seconds=sum(r['seconds'] for r in route_trials))
        return labels, metrics


def configure():
    online = base.configure()
    recovery.training_records = selection.training_records = training_records
    online.HSLBuffer = DecisionCoverageBuffer
    online.mixed_records = recovery.mixed_records
    online.ODMController = BufferValueController
    online.OnlineHSLTrainer = TextCraftDCHLTrainer
    return online


if __name__ == '__main__':
    configure().main()
