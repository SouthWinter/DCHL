"""PlanCraft Data Agent with verified inventory and action support."""
import train_plancraft_decision_coverage_structural_feedback as base
import train_plancraft_decision_coverage as core
from agent_data_agent import DataAgentBuffer
from train_agent_data_agent import DataAgentMixin


class Trainer(DataAgentMixin,core.DecisionCoverageTrainer):
    pass


if __name__=='__main__':
    online=base.configure()
    online.HSLBuffer=DataAgentBuffer
    online.OnlineHSLTrainer=Trainer
    online.main()
