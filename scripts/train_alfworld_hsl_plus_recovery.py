"""HSL FIFO/quality sampler plus the unchanged DCHL continuation procedure."""
import train_alfworld_verified_recovery as parent
from alfworld_continuation_compression import choose_jobs, evidence_entry
from alfworld_hsl_plus_recovery import HSLRecoveryBuffer, mixed_records


class HSLRecoveryTrainer(parent.RecoveryTrainer):
    collector_script = 'collect_alfworld_continuation_compression.py'


def configure():
    parent.choose_jobs = choose_jobs
    parent.recovery_entry = evidence_entry
    parent.online.HSLBuffer = HSLRecoveryBuffer
    parent.online.mixed_records = mixed_records
    parent.online.OnlineHSLTrainer = HSLRecoveryTrainer
    return parent.online


if __name__ == '__main__':
    configure().main()
