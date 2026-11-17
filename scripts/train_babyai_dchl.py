"""Existing decision coverage, bounded feedback and verified continuation."""
import train_textcraft_dchl as parent
import train_babyai_hsl as baby
from babyai_adapter import DATA, BabySession, action_records, training_records


def compress_successful_suffix(session, example, original, cut):
    best = original
    keep = list(range(len(original['trajectory'])))
    checks = 0
    for index in reversed(keep[cut:]):
        if checks >= 24:
            break
        trial_indices = [i for i in keep if i != index]
        trial = session.replay(example, [original['trajectory'][i]['action'] for i in trial_indices])
        checks += 1
        if (trial['won'] and len(trial['trajectory']) == len(trial_indices)
            and all(t['intent'] == original['trajectory'][i]['intent'] for i,t in zip(trial_indices,trial['trajectory']))):
            keep, best = trial_indices, trial
    assert best['won'] and best['trajectory'][:cut] == original['trajectory'][:cut]
    return best, checks


class BabyAIDCHLTrainer(parent.TextCraftDCHLTrainer):
    def make_recovery_goal(self, session, example, original, cut, tokenizer, description):
        best, checks = compress_successful_suffix(session, example, original, cut)
        records = action_records(best['messages'][:-1], tokenizer)
        mask = [i >= cut and t['valid'] for i,t in enumerate(best['trajectory'])]
        assert any(mask)
        goal = dict(instruction=description, task_type=example['goal_spec']['kind'],
            prefix_length=len(records), action_loss_mask=mask, records=records)
        return goal, dict(checks=checks, original_validity_fallback=False,
            selected=[i for i, enabled in enumerate(mask) if enabled])


def configure():
    baby.configure()
    parent.DATA = DATA
    parent.TextSession = BabySession
    parent.action_records = action_records
    parent.training_records = training_records
    parent.compress_successful_suffix = compress_successful_suffix
    online = parent.configure()
    online.OnlineHSLTrainer = BabyAIDCHLTrainer
    return online


if __name__ == '__main__':
    configure().main()
