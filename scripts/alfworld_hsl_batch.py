"""Mixed trajectory batches with the existing VERL tokenizer and masks."""
from collections import defaultdict
import copy
import random
import torch

import pandas as pd
from verl.utils.dataset.multiturn_sft_dataset import (
    MultiTurnSFTDataset, extract_system_prompt_and_generation,
)

from alfworld_hsl_buffer import training_records


class InMemorySFTDataset(MultiTurnSFTDataset):
    def _process_single_message(self, index, message, *args, **kwargs):
        message=copy.deepcopy(message)
        start=message.pop('loss_char_start',None)
        result=super()._process_single_message(index,message,*args,**kwargs)
        if start is None or not message.get('loss_on_this_turn',True):
            return result
        ids,mask,attention,inputs=result
        # Decode the actual per-turn tokens, then verify exact retokenization.
        # This aligns the character span with the installed chat template/BPE.
        text=self.tokenizer.decode(ids.tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False)
        encoded=self.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        if encoded['input_ids']!=ids.tolist():
            raise ValueError('Action span tokenization does not match actual message')
        content=message['content'];position=text.find(content)
        adjusted_start=int(start)
        # Llama's chat template trims message-edge whitespace. Preserve the
        # raw reply in the record, but align its span to the rendered content.
        if position<0:
            position=text.find(content.strip())
            adjusted_start-=len(content)-len(content.lstrip())
        if position<0 or not 0<=int(start)<len(content) or adjusted_start<0:
            raise ValueError('Action span not found in actual message')
        boundary=position+adjusted_start
        keep=torch.tensor([a>=boundary for a,b in encoded['offset_mapping']],device=mask.device)
        mask=mask*keep
        if not mask.any():raise ValueError('Action span has no supervised tokens')
        return ids,mask,attention,inputs

    def __init__(self, rows, tokenizer, config):
        self.rows = rows
        self.sampling_statistics = config.get('sampling_statistics', False)
        super().__init__([], tokenizer, config)

    def __getitem__(self, index):
        while True:
            try:
                item = super().__getitem__(index)
                break
            except ValueError as exc:
                # Preserve initial instructions, initial observation, latest
                # context and target when a full-response hindsight exceeds cap.
                messages=self.messages[index]
                if ('is larger than' not in str(exc) or len(messages)<=5
                    or 'loss_char_start' not in messages[-1]):
                    raise
                self.messages[index]=messages[:3]+messages[5:]
        if self.sampling_statistics:
            item['sampling_draw_id'] = torch.full_like(item['input_ids'], self.rows[index].get('sampling_draw_id', -1))
        return item

    def _read_files_and_process(self):
        self.dataframe = pd.DataFrame(self.rows)
        self.messages = [r[self.messages_key] for r in self.rows]
        self.tools = None
        self.enable_thinking = [r.get(self.enable_thinking_key, False) for r in self.rows]
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)


def group_demonstrations(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source_episode_id']].append(row)
    for records in grouped.values():
        records.sort(key=lambda r: r['step_index'])
        if [r['step_index'] for r in records] != list(range(1, len(records) + 1)):
            raise ValueError('original demonstration is not a complete ordered trajectory')
    return dict(grouped)


def epoch_task_batch(demonstrations, step, batch_size, seed):
    """Use every original task once per epoch, independently of rollout sampling."""
    tasks = sorted(demonstrations)
    orders, selected = {}, []
    for position in range((step - 1) * batch_size, step * batch_size):
        epoch, index = divmod(position, len(tasks))
        if epoch not in orders:
            orders[epoch] = list(tasks)
            random.Random(seed + epoch).shuffle(orders[epoch])
        selected.append(orders[epoch][index])
    return selected


def mixed_records(demonstrations, buffer, rng, trajectories=32, hindsight_fraction=.3, original_tasks=None, goal_logits=None):
    if trajectories < 2 or not 0 < hindsight_fraction < 1:
        raise ValueError('mixed batch requires at least two trajectories and a fraction in (0,1)')
    # Estimate the two expectations independently. Mixture controls loss weight,
    # not how many original demonstrations survive the addition of hindsight.
    selected = buffer.sample(trajectories, goal_logits=goal_logits)
    gt_count = trajectories
    gt_fraction = 1.0 - hindsight_fraction if selected else 1.0
    output, provenance = [], []
    if original_tasks is None:
        original_tasks = rng.choices(sorted(demonstrations), k=gt_count)
    if len(original_tasks) != gt_count:
        raise ValueError('original task batch differs from trajectory batch size')
    for slot, task in enumerate(original_tasks):
        source = demonstrations[task]
        for row in source:
            record = copy.deepcopy(row)
            record['trajectory_action_coefficient'] = gt_fraction / gt_count / len(source)
            record['mixed_trajectory_draw'] = ('original', slot)
            output.append(record)
        provenance.append({'source': 'original', 'task': task, 'actions': len(source)})
    for draw, sample in enumerate(selected):
        rollout, goal = sample['rollout'], sample['goal']
        rows = training_records(rollout, goal, hindsight_fraction, len(selected))
        for row in rows:
            row['sampling_draw_id'] = draw
            row['mixed_trajectory_draw'] = ('hindsight', draw)
        # Zero-loss current actions need no forward pass. Their observations and
        # actions remain in the inputs of all subsequent supervised records.
        output.extend(r for r in rows if r['action_supervised'])
        provenance.append({'source': 'hindsight', 'task': rollout['gamefile'],
            'sampling_draw_id': draw,
            'goal_task_type': goal['task_type'],
            'goal': goal['instruction'], 'prefix_length': goal['prefix_length'],
            'original_request_bound':bool(goal.get('original_request_bound',False)),
            'canonical_instruction':goal.get('canonical_instruction'),
            'supervised_actions': goal['relevant_actions'],
            'policy_version': sample['policy_version'],
            'cache_source_id': sample.get('cache_source_id'),
            'sample_probability': sample['sample_probability']})
        if 'source_attempt_id' in rollout:
            provenance[-1]['source_attempt_id'] = rollout['source_attempt_id']
    rng.shuffle(output)
    return output, provenance


def rank_records(rows, rank, world_size, microbatch):
    """Equal microbatch counts on every FSDP rank; padding has exactly zero loss."""
    rows = list(rows)
    multiple = world_size * microbatch
    while len(rows) % multiple:
        dummy = copy.deepcopy(rows[0])
        for message in dummy['messages']:
            message['loss_on_this_turn'] = False
        rows.append(dummy)
    return rows[rank::world_size]
