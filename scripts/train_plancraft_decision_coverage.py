"""DCHL with native manufacturing witnesses and inventory-aligned references."""
import train_plancraft_continuation as parent
from train_alfworld_direct_candidate import DirectCandidateTrainer
from dchl_dataset_support import FixedSupportMixin, install, with_source_metadata
from plancraft_hindsight import training_records
from plancraft_aligned_reference import choose_jobs


class DecisionCoverageTrainer(FixedSupportMixin, parent.ContinuationTrainer, DirectCandidateTrainer):
    pass


def configure():
    parent.choose_jobs = choose_jobs
    return install(parent, DecisionCoverageTrainer, with_source_metadata(training_records))


if __name__ == '__main__':
    configure().main()
