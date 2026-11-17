"""Use all execution-verified necessary actions as PlanCraft decision support."""
from plancraft_recovery import evidence_labels as original_labels


def evidence_labels(row):
    labels=original_labels(row)
    for goal in labels['goals']:
        support=[i for i,keep in enumerate(goal['action_loss_mask']) if keep]
        assert support and len(support)==goal['relevant_actions']
        goal['feedback_steps']=support
    return labels


def configure():
    import train_plancraft_decision_coverage as base
    online=base.configure()
    base.parent.evidence_labels=evidence_labels
    return online


if __name__=='__main__':
    configure().main()
