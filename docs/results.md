# Result provenance

`results/main.json` records numerator and denominator for exact reconstruction; README reports success percentages rounded to two decimals. Differences are percentage points.

- Main online seeds: ALFWorld 20260822; PlanCraft and BabyAI 42.
- Main primary budgets: ALFWorld 75 updates / 1,350 trajectories; PlanCraft 89 / 1,602; BabyAI 25 / 450.
- Evaluation: ALFWorld 140 seen + 134 unseen; PlanCraft 480 feasible tasks; BabyAI AgentGym-derived native 90-task panel.
- Qwen ALFWorld uses the completed rebuilt-common-start comparison, including HSL 89.05% and DCHL 93.07%.
- Qwen PlanCraft's selected HSL/DCHL pair comes from the original SFT (both start at 88.75%). DynamixSFT/Data Agent use a rebuilt SFT with the same aggregate success. The rebuilt HSL/DCHL comparison is 90.21% / 87.50%, rather than the selected original pair's 89.38% / 90.42%.
- Original Qwen PlanCraft weights were retired. Code and recipes support rerunning the training procedure, not byte-identical recovery of those weights.
- Main results report one online seed; online seed changes do not represent independent SFT replications. The development process repeatedly used the evaluation panels.
- Primary trajectory budgets are matched; verified continuation attempts and actual compute are additional and method-dependent.
- HSL plus continuation is an additional control still pending at this release. No result is inferred for it.

This release does not bundle model checkpoints, training trajectories, private documents, or archived exploratory runs. `docs/source_manifest.json` links released modules to research-source hashes and records portability changes. Validation of this export is reported in `docs/release_validation.json`.
