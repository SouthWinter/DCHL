"""PlanCraft DynamixSFT with public decision-depth arms."""
import train_plancraft_decision_coverage_structural_feedback as base
import train_plancraft_decision_coverage as core
from agent_dynamix_transfer import DynamixTransferBuffer
from train_agent_dynamix_transfer import DynamixProbeMixin


class PlanCraftBuffer(DynamixTransferBuffer):
    dataset = 'plancraft'


class Trainer(DynamixProbeMixin, core.DecisionCoverageTrainer):
    pass


if __name__ == '__main__':
    online = base.configure()
    online.HSLBuffer = PlanCraftBuffer
    online.OnlineHSLTrainer = Trainer
    online.main()
