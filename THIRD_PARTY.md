# Third-party components

External code is fetched separately, not copied into this repository as an uncredited dependency.

| Component | Source | Role |
|---|---|---|
| verl | https://github.com/volcengine/verl | Full-parameter FSDP training; Apache-2.0 upstream |
| PlanCraft | https://github.com/gautierdag/plancraft | Native crafting transitions, oracle traces and evaluation tasks |
| AgentGym | https://github.com/WooooDyy/AgentGym | BabyAI macro-action interface and TextCraft compatibility dependency |
| MiniGrid / BabyAI | https://github.com/Farama-Foundation/Minigrid | Native instruction semantics and verifier |
| ALFWorld | https://github.com/alfworld/alfworld | Household environment |
| ALFWorld public evaluator/data | https://huggingface.co/datasets/zhangdw/alfworld | Public evaluation engine and task manifests |
| ETO demonstrations | https://huggingface.co/datasets/agent-eto/eto-sft-trajectory | Public ALFWorld demonstration source |
| AgentTraj-L / AgentEval | https://huggingface.co/AgentGym | BabyAI demonstration and evaluation source |

Pinned Git revisions are in `tools/setup_external.py`. The verl patch modifies upstream files under their existing Apache-2.0 notices; obtain the base repository and its LICENSE before applying it. Model weights and chat templates remain governed by Meta Llama and Qwen's respective terms. Dataset use is governed by its source license. No license for third-party assets is replaced by this repository.
