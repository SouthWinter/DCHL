"""Existing Beta-PPO selector on BabyAI with the actual model vocabulary."""
import json
import os
from pathlib import Path
import train_babyai_dchl as base
from agent_data_agent import DataAgentBuffer
from train_agent_data_agent import DataAgentMixin


class BabyAIDataAgentBuffer(DataAgentBuffer):
    pass


class Trainer(DataAgentMixin, base.BabyAIDCHLTrainer):
    pass


def configure():
    model = Path(os.environ['ALFWORLD_INITIAL_MODEL_PATH'])
    BabyAIDataAgentBuffer.vocab_size = json.loads((model/'config.json').read_text())['vocab_size']
    online = base.configure()
    online.HSLBuffer = BabyAIDataAgentBuffer
    online.OnlineHSLTrainer = Trainer
    return online


if __name__ == '__main__':
    configure().main()
