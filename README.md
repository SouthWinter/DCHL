# DCHL: Decision-Coverage Hindsight Learning

DCHL learns from multi-goal agent experience by allocating training opportunities according to the execution decisions supporting each goal. Goals from one trajectory often share decisions; DCHL divides their shared credit and reduces repeated exposure while retaining complementary evidence.

This repository contains the research implementation for **ALFWorld, PlanCraft, and BabyAI**, with Llama-3.2-1B-Instruct and Qwen3-4B-Instruct-2507 recipes. It includes local agent adaptations of **HSL, DynamixSFT, and Data Agent**, full-parameter training, environment adapters, evaluation, and CPU regression tests.

## Start here

```bash
# The allocation example needs only Python; it calls the actual method class.
python tools/demo_coverage.py
python -m pip install pytest
python -m pytest -q
```

The example has supports `{a,b}`, `{a,b}`, `{a,c}`. Initial goal probabilities are `5/18, 5/18, 4/9`; reserving the third goal makes the next structural priorities equal.

For full training, follow [installation and data preparation](docs/installation.md), then [reproduction](docs/reproduction.md). Download model weights and upstream datasets separately. The released launchers have been syntax-checked and dry-run validated; this export has not undergone a fresh end-to-end GPU reproduction.

## Method

For goal `g` in source trajectory `τ`, the structural credit is

```text
C(g | τ) = Σ[j in support(g)] 1 / (multiplicity(τ,j) × (1 + exposure(τ,j)))
```

Quality and bounded learning feedback modulate this credit. Source allocation controls the contribution of each trajectory; goal selection accounts for both committed exposure and draws already reserved in the current batch. Original demonstrations, hindsight supervision, and execution-verified original-request continuations form the full training recipe.

See [method and implementation map](docs/method.md) for retention, feedback, supervision masks, and environment-specific support definitions.

## Main results

Complete-task success rate (%), one online seed. `Dyn.` is DynamixSFT; `DA` is Data Agent.

| Model | Dataset | SFT | HSL | Dyn. | DA | DCHL |
|---|---|---:|---:|---:|---:|---:|
| Llama 1B | ALFWorld | 72.99 | 87.96 | 88.69 | 85.77 | **91.61** |
| Llama 1B | PlanCraft | 65.21 | 75.00 | 75.00 | 75.63 | **77.92** |
| Llama 1B | BabyAI | 80.00 | 75.56 | **85.56** | 82.22 | 84.44 |
| Qwen 4B | ALFWorld | 74.82 | 89.05 | **93.07** | 91.97 | **93.07** |
| Qwen 4B | PlanCraft† | 88.75 | 89.38 | 86.46† | 87.50† | **90.42** |
| Qwen 4B | BabyAI | 82.22 | 85.56 | 86.67 | 86.67 | **91.11** |

Evaluation sizes: ALFWorld 274, PlanCraft 480, BabyAI 90. † The selected Qwen PlanCraft HSL/DCHL pair uses the original SFT checkpoint. The external selectors use a rebuilt SFT with the same aggregate starting score but different weights. In the later rebuilt-start HSL/DCHL pair, success is **90.21% / 87.50%**. This row is not a matched four-way comparison; see [result provenance](docs/results.md). Regenerating SFT does not recreate a retired checkpoint merely because its aggregate score matches.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/alfworld_decision_coverage.py` | Shared decision credit and exposure |
| `scripts/alfworld_batch_selection.py` | Hierarchical source/goal allocation |
| `scripts/*adapter.py`, `*hindsight.py` | Environment transitions and verified goals |
| `scripts/train_*.py` | DCHL, comparator and shared training implementations |
| `configs/` | Six model–environment online recipes |
| `tools/` | Setup, data download, training launcher and CPU example |
| `patches/verl-sft.patch` | Required supervised-training framework changes |
| `tests/` | Coverage, exposure, replay, scheduling and action-contract invariants |
| `results/main.json` | Selected main-table scores and denominators |
| `docs/` | Installation, reproduction, method, provenance and dependencies |

Historical module names are retained where they are part of the tested dependency chain. `train_alfworld_direct_candidate.py` executes one optimizer update; inheriting a historical helper does not enable the older two-update selection experiment.

## References and licensing

- [HSL](https://arxiv.org/abs/2607.04235)
- [DynamixSFT](https://aclanthology.org/2026.findings-acl.1972/)
- [Data Agent](https://proceedings.mlr.press/v306/yang26bq.html)

Third-party code and datasets retain their upstream licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). A project-wide license for the original DCHL code has not yet been selected. Public visibility alone does not grant an additional license.
