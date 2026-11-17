"""ALFWorld Data Agent with the validated action-contract candidate carrier."""
import train_alfworld_direct_candidate as base
from agent_data_agent import DataAgentBuffer
from train_agent_data_agent import DataAgentMixin


class Trainer(DataAgentMixin, base.DirectCandidateTrainer):
    pass


if __name__=='__main__':
    base.parent.choose_jobs=base.choose_jobs
    base.parent.recovery_entry=base.evidence_entry
    online=base.parent.online
    online.HSLBuffer=DataAgentBuffer
    online.mixed_records=base.parent.mixed_records
    online.ODMController=base.parent.BufferValueController
    online.OnlineHSLTrainer=Trainer
    online.main()
