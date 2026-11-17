# Method and implementation

## Execution evidence

A source is identified by policy version, environment instance, and rollout ID. Each verified hindsight goal contains an instruction, a trajectory prefix, a supervision mask, a quality weight, and `feedback_steps` identifying execution support. Shared physical positions do not imply identical goal-conditioned gradients.

## Source and goal allocation

`alfworld_stable_evidence.py` retains maximal support certificates within each source and preserves supported task types. It computes bounded source credit. `alfworld_batch_selection.py` keeps the underlying source rollouts and constructs the two-level draw: task-type slots, source budgets, then goals within each source.

`alfworld_decision_coverage.py` computes multiplicity over the source's candidate goals. Each supported decision receives credit inversely proportional to multiplicity and exposure. The current batch's reserved draws count immediately. `commit_proposal()` updates persistent exposure only for the selected update; `state_dict()` retains it across checkpoint recovery.

`alfworld_soft_rehearsal.py`, `alfworld_bounded_hsl.py`, and `alfworld_stable_evidence.py` implement bounded, support-aware family feedback from observed training loss. This is relative loss feedback, not an estimate of counterfactual learning gain.

## Supervised update

The main mixture draws 32 original demonstrations (coefficient 0.70) and 32 hindsight goals (0.30). When available, four verified continuation records contribute an additional coefficient 0.05. Hindsight targets use action masks and prefix-length normalization. Continuations supervise the accepted suffix and normalize by its supervised action count.

`train_alfworld_direct_candidate.py` prepares the selected batch and executes **one** `train_batch`. Its parent module contains historical utilities, but the direct trainer overrides the update-selection operation. ALF-prefixed utilities are shared infrastructure for all three environments.

## Environment adapters

| Environment | Goal verification / support | Entry point |
|---|---|---|
| ALFWorld | Successful action responses; execution support and supervision masks are distinct | `train_alfworld_decision_coverage` |
| PlanCraft / Llama | Native inventory transitions, deletion-verified necessary actions; structural feedback families | `train_plancraft_decision_coverage_structural_feedback` |
| PlanCraft / Qwen selected main recipe | Manufacturing-action evidence and item-family feedback | `train_plancraft_decision_coverage` |
| BabyAI | Native instruction predicates, object-aware deletion replay, structural families | `train_babyai_dchl` |

The BabyAI trainer reuses generic TextCraft trajectory scaffolding but installs BabyAI execution, labeling, serialization, and continuation verification before training. Upstream TextCraft files are still an import dependency in this release.

## Comparators

HSL uses FIFO goal retention and quality-weighted sampling with local environment-backed relabeling. The local HSL adaptation does not invoke the paper's auxiliary 70B relabeler. DynamixSFT and Data Agent use the published selector ideas inside the same agent collection pipeline; these are locally implemented adaptations. Their complete recipes include the documented continuation interface. Compare common SFT checkpoints and primary collection budgets, and separately account for extra generation, replay and supervised tokens.
