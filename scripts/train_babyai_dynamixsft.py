"""Published DynamixSFT arm policy on BabyAI native instruction families."""
import train_babyai_dchl as base
from agent_dynamix_transfer import DynamixTransferBuffer
from train_agent_dynamix_transfer import DynamixProbeMixin


class BabyAIDynamixBuffer(DynamixTransferBuffer):
    def arm(self, goal):
        return goal['task_type']


class Trainer(DynamixProbeMixin, base.BabyAIDCHLTrainer):
    pass


def configure():
    online = base.configure()
    online.HSLBuffer = BabyAIDynamixBuffer
    online.OnlineHSLTrainer = Trainer
    return online


if __name__ == '__main__':
    configure().main()
