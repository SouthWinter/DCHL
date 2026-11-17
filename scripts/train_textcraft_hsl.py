"""TextCraft collector with the existing online HSL optimizer and objective."""
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import train_alfworld_hsl as online
import alfworld_hsl_batch as batch
from alfworld_data_schedule import round_tasks
from dchl_dataset_support import FixedSupportMixin
from textcraft_adapter import TextSession, context_messages, training_records, MAX_STEPS
from textcraft_hindsight import relabel

HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(endpoint, messages, tokenizer, seed):
    body = dict(model='qwen3-4b', messages=context_messages(messages, tokenizer),
        temperature=1., top_p=1., max_tokens=128, seed=seed)
    req = urllib.request.Request(endpoint+'/v1/chat/completions', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with HTTP.open(req, timeout=180) as response:
        return json.load(response)['choices'][0]['message']['content'] or ''


class TextCraftTrainer(FixedSupportMixin, online.OnlineHSLTrainer):
    def collect_round(self, step, tasks, rng, version, root):
        started = time.monotonic()
        directory = root/'rounds'/f'{step:05d}'
        directory.mkdir(parents=True, exist_ok=True)
        selected = round_tasks(tasks, int(self.config.hsl.rollouts_per_step), int(self.config.trainer.seed), step, rng, True)
        (directory/'tasks.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in selected))
        if not hasattr(self, 'textcraft_pool'):
            self.textcraft_pool = ThreadPoolExecutor(max_workers=int(self.config.hsl.workers))
            self.textcraft_local = threading.local()
            self.textcraft_sessions = []
            self.textcraft_targets = {e['goal'] for e in tasks}
        tokenizer = self.model_config.tokenizer

        def episode(pair):
            index, example = pair
            if not hasattr(self.textcraft_local, 'session'):
                self.textcraft_local.session = TextSession()
                self.textcraft_sessions.append(self.textcraft_local.session)
            session = self.textcraft_local.session
            session.reset(example)
            description = session.description
            for turn in range(MAX_STEPS):
                answer = request(self.config.hsl.endpoint, session.messages, tokenizer,
                    int(self.config.trainer.seed)+step*100003+index*101+turn)
                session.step(answer)
                if session.done:
                    break
            trajectory, won = list(session.trajectory), session.won
            hindsight = relabel(example, trajectory, tokenizer, version, session, self.textcraft_targets)
            hindsight['rollout_id'] = index
            return dict(rollout_id=index, id=example['id'], status='complete', won=won,
                score=100.*won, description=description, trajectory=trajectory, hindsight=hindsight)

        rows = list(self.textcraft_pool.map(episode, enumerate(selected)))
        (directory/'rollouts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        metrics = dict(collection_seconds=time.monotonic()-started, rollout_count=len(rows),
            successes=sum(row['won'] for row in rows),
            generated_goals=sum(len(row['hindsight']['goals']) for row in rows),
            relevance_checks=sum(row['hindsight']['relevance_checks'] for row in rows),
            multi_goal_sources=sum(len(row['hindsight']['goals']) > 1 for row in rows))
        return [row['hindsight'] for row in rows], metrics

    def close_owned_service(self):
        if hasattr(self, 'textcraft_pool'):
            self.textcraft_pool.shutdown(wait=True)
            for session in self.textcraft_sessions:
                session.close()
        super().close_owned_service()


def configure():
    batch.training_records = training_records
    online.OnlineHSLTrainer = TextCraftTrainer
    return online


if __name__ == '__main__':
    configure().main()
