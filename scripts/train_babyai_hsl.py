"""BabyAI collector on the existing HSL optimizer, FIFO and loss."""
import train_textcraft_hsl as base
from babyai_adapter import BabySession, context_messages, relabel, training_records, MAX_STEPS


def configure():
    if not getattr(base.TextCraftTrainer, '_babyai_trace_errors', False):
        original_collect = base.TextCraftTrainer.collect_round
        def collect_with_trace(self, *args, **kwargs):
            try:
                return original_collect(self, *args, **kwargs)
            except Exception:
                import traceback
                traceback.print_exc()
                raise
        base.TextCraftTrainer.collect_round = collect_with_trace
        base.TextCraftTrainer._babyai_trace_errors = True
    base.TextSession = BabySession
    base.context_messages = context_messages
    base.relabel = relabel
    base.training_records = training_records
    base.MAX_STEPS = MAX_STEPS
    return base.configure()


if __name__ == '__main__':
    configure().main()
