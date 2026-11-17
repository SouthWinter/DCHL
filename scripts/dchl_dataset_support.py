"""Serialization and checkpoint support shared by the two DCHL transfers."""
import json
import os
from pathlib import Path
import shutil


def with_source_metadata(serializer):
    def records(rollout, goal, mixture=1., sampled_trajectories=1):
        rows = serializer(rollout, goal, mixture, sampled_trajectories)
        for index, row in enumerate(rows):
            row.update(policy_version=rollout['policy_version'],
                       source_episode_id=rollout['gamefile'], step=index)
        return rows
    return records


class FixedSupportMixin:
    def __init__(self, config):
        super().__init__(config)
        template = Path(os.environ['ALFWORLD_FIXED_CHAT_TEMPLATE']).read_text()
        self.model_config.tokenizer.chat_template = template
        original_save = self.ckpt_handler.save_checkpoint

        def save(step):
            original_save(step)
            if self.rank == 0:
                source = Path(os.environ['ALFWORLD_INITIAL_MODEL_PATH'])
                target = Path(config.trainer.default_local_dir)/f'global_step_{step}/huggingface'
                for name in ('config.json', 'generation_config.json', 'tokenizer.json',
                             'tokenizer_config.json', 'special_tokens_map.json'):
                    if (source/name).exists():
                        shutil.copy2(source/name, target/name)
                    elif name == 'special_tokens_map.json':
                        # Some base Qwen bundles omit this optional file; the
                        # checkpoint tokenizer already generated it correctly.
                        assert (target/name).exists()
                    else:
                        raise FileNotFoundError(source/name)
                path = target/'tokenizer_config.json'
                value = json.loads(path.read_text())
                value['chat_template'] = template
                path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
                source_generation = json.loads((source/'generation_config.json').read_text())
                target_generation = json.loads((target/'generation_config.json').read_text())
                assert target_generation['eos_token_id'] == source_generation['eos_token_id']
        self.ckpt_handler.save_checkpoint = save


def install(parent, trainer, records):
    import alfworld_batch_selection as selection
    from alfworld_decision_coverage import DecisionCoverageBuffer
    parent.base.batch.training_records = records
    parent.recovery.training_records = records
    selection.training_records = records
    online = parent.base.online
    online.HSLBuffer = DecisionCoverageBuffer
    online.mixed_records = parent.recovery.mixed_records
    online.ODMController = parent.BufferValueController
    online.OnlineHSLTrainer = trainer
    return online
