"""Use the existing native-agent evaluator with BabyAI's frozen interface."""
import evaluate_textcraft as base
from babyai_adapter import DATA, BabySession, context_messages, MAX_STEPS

if __name__ == '__main__':
    base.DATA = DATA
    base.TextSession = BabySession
    base.context_messages = context_messages
    base.MAX_STEPS = MAX_STEPS
    base.main()
