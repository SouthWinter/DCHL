"""Full necessary-action support with execution-structure feedback families."""
import json
from train_plancraft_decision_coverage_required_actions import evidence_labels as full_labels
from plancraft_execution_structure import structure


def evidence_labels(row):
    labels=full_labels(row)
    for goal in labels['goals']:
        actions=[step['action'] for step,keep in zip(row['trajectory'],goal['action_loss_mask']) if keep]
        goal['object_type']=json.dumps(structure(actions),ensure_ascii=False,separators=(',',':'))
    return labels


def configure():
    import train_plancraft_decision_coverage as base
    online=base.configure()
    base.parent.evidence_labels=evidence_labels
    return online


if __name__=='__main__':configure().main()
