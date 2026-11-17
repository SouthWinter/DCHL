"""Single-update training with execution-decision coverage allocation."""
from train_alfworld_direct_candidate import DirectCandidateTrainer, parent, choose_jobs, evidence_entry
from alfworld_decision_coverage import DecisionCoverageBuffer


if __name__ == '__main__':
    parent.choose_jobs = choose_jobs
    parent.recovery_entry = evidence_entry
    online = parent.online
    online.HSLBuffer = DecisionCoverageBuffer
    online.mixed_records = parent.mixed_records
    online.ODMController = parent.BufferValueController
    online.OnlineHSLTrainer = DirectCandidateTrainer
    online.main()
