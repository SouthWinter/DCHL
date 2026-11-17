"""Freeze the public split and rebuild native demonstration observations."""
import copy
import json
from pathlib import Path
import time
import pandas as pd
from transformers import AutoTokenizer
from babyai_adapter import ROOT, DATA, BabySession, action_records, CONTEXT, all_levels, encode
from screen_babyai import demonstration_actions


def main():
    DATA.mkdir(exist_ok=True)
    started = time.monotonic()
    source = ROOT/'data/babyai_screen_v1/source'
    train = json.loads((source/'babyai_train.json').read_text())
    test = json.loads((source/'babyai_test.json').read_text())
    assert not {x['item_id'] for x in train} & {x['item_id'] for x in test}
    tokenizer = AutoTokenizer.from_pretrained(__import__('os').environ['DCHL_BASE_MODEL'], local_files_only=True)
    tokenizer.chat_template = tokenizer.chat_template.replace('strftime_now("%d %b %Y")', '"01 Nov 2026"')
    (DATA/'fixed_chat_template.jinja').write_text(tokenizer.chat_template)
    session = BabySession()
    examples, sft, online, excluded = [], [], [], []
    invalid = order_fixed = 0
    max_tokens = 0
    for i, row in enumerate(train):
        example = dict(id=row['item_id'])
        result = session.replay(example, demonstration_actions(row))
        idx = int(example['id'].split('_')[-1])
        example.update(goal=result['description'], depth=idx%40+1, level=all_levels[idx%40+1],
                       goal_spec=session.goal_spec)
        published = row['conversations'][2]['value'].split('\n')[0].removeprefix('Your goal: ')
        order_fixed += published != result['description']
        result.update(id=example['id'], example=example)
        (DATA/'replayed').mkdir(exist_ok=True)
        (DATA/'replayed'/f"{example['id']}.json").write_text(json.dumps(result, ensure_ascii=False))
        if not result['won']:
            excluded.append(dict(id=example['id'], reason='published demonstration fails under native terminal handling', steps=len(result['trajectory'])))
            continue
        examples.append(example)
        messages = copy.deepcopy(result['messages'][:-1])
        mask = iter(t['valid'] for t in result['trajectory'])
        for m in messages:
            m['loss_on_this_turn'] = next(mask) if m['role']=='assistant' else False
        invalid += sum(not t['valid'] for t in result['trajectory'])
        n = len(tokenizer.apply_chat_template(messages, tokenize=True))
        assert n <= CONTEXT, (example['id'], n)
        max_tokens = max(n, max_tokens)
        sft.append(dict(messages=messages, enable_thinking=False, source_episode_id=example['id']))
        records = action_records(messages, tokenizer)
        for j, (record, transition) in enumerate(zip(records, result['trajectory'])):
            record['messages'][-1]['loss_on_this_turn'] = transition['valid']
            record.update(source_episode_id=example['id'], step_index=j+1)
        online.extend(records)
        if (i+1)%100==0:
            print('prepared', i+1, 'successful', len(examples), flush=True)
    tests = []
    # IDs and level metadata only: TEST environments are not executed here.
    for row in test:
        idx = int(row['item_id'].split('_')[-1])
        tests.append(dict(id=row['item_id'], goal='native mission reconstructed on reset', depth=idx%40+1, level=all_levels[idx%40+1]))
    session.close()
    (DATA/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples))
    (DATA/'test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tests))
    pd.DataFrame(sft).to_parquet(DATA/'train.parquet', index=False)
    pd.DataFrame(online).to_parquet(DATA/'hsl_demonstrations.parquet', index=False)
    result = dict(status='PASS', public_train=810, train_sources=len(examples), test=90,
        excluded=excluded, original_invalid_actions_context_only=invalid, order_prompts_restored=order_fixed,
        max_tokens=max_tokens, action_records=len(online), template_date='01 Nov 2026',
        seconds=time.monotonic()-started, native_terminal=True, test_execution=0,
        source='AgentGym/AgentTraj-L and AgentGym/AgentEval public BabyAI IDs',
        common_protocol='native instruction order, stop macro at native termination/truncation, singleton action JSON accepted')
    (DATA/'manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__=='__main__':
    main()
