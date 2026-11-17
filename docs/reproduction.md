# Reproduction and execution

## 1. Build one common SFT checkpoint

All online methods in a comparison must load the same SFT **weights**, tokenizer and template. The `scripts/run_alfworld_sft_stage_4gpu.sh` backend is configurable for one to eight GPUs despite its historical filename. Its positional arguments are:

```text
DATA.parquet TARGET_UPDATES RUN_DIR CHECKPOINT_DIR {disable|auto} LOSS_WEIGHT_KEY GPU_LIST [Hydra overrides...]
```

The preserved demonstration builders supply the training records. Historical common-SFT budgets were ALFWorld 75 updates, PlanCraft Llama 63 / Qwen 66, and BabyAI 27. Batch layout, template and masking are part of the initialization protocol. Exact retired checkpoints are not recoverable from aggregate evaluation scores.

For example, BabyAI uses three epochs over 810 demonstrations, global batch 84, 27 updates, learning rate 2e-5, cosine schedule with 3% warmup, zero weight decay, and seed 42:

```bash
source env.sh
export ALFWORLD_INITIAL_MODEL_PATH="$DCHL_BASE_MODEL"
export ALFWORLD_FIXED_CHAT_TEMPLATE="$DCHL_ROOT/data/babyai_v1/fixed_chat_template.jinja"
export ALFWORLD_SFT_MODULE=train_babyai_sft ALFWORLD_SFT_LR=2e-5
export ALFWORLD_MODEL_REMOVE_PADDING=false ALFWORLD_ISOLATE_SAMPLES=1
export ALFWORLD_TRAIN_MICRO_BATCH_SIZE=6 ALFWORLD_SAVE_FREQ=27
bash scripts/run_alfworld_sft_stage_4gpu.sh \
  data/babyai_v1/train.parquet 27 runs/babyai_sft/train \
  runs/babyai_sft/checkpoints disable '' 0,1,2,3,4,5,6 \
  data.train_batch_size=84 data.max_length=4096 data.max_token_len_per_gpu=4096 \
  trainer.seed=42 trainer.total_epochs=3 optim.lr_scheduler_type=cosine \
  optim.lr_warmup_steps_ratio=0.03 optim.weight_decay=0 \
  'checkpoint.save_contents=[optimizer,extra,hf_model]' \
  'checkpoint.load_contents=[optimizer,extra]'
```

Choose the physical batch layout for the allocated device count explicitly; do not silently alter effective supervision when moving between layouts. The example uses the historical seven-device layout. Save and protect the common SFT before branching.

## 2. Run an online method

```bash
source env.sh
python tools/train.py --dataset babyai --model llama --method dchl \
  --sft runs/babyai_sft/checkpoints/global_step_27/huggingface \
  --run runs/babyai_llama_dchl --gpus 0,1,2,3,4,5,6 --dry-run
# Review the generated config, then repeat without --dry-run.
```

Supported datasets: `alfworld`, `plancraft`, `babyai`; model recipes: `llama`, `qwen`; methods: `hsl`, `dchl`, `dynamixsft`, `data_agent`. All paths and devices are explicit. Training does not stop other users' processes. The launcher requires unoccupied selected GPUs; use dedicated allocations.

| Dataset | Updates | Primary rollouts | Max action steps | Online LR |
|---|---:|---:|---:|---:|
| ALFWorld | 75 | 1,350 | 40 | 2e-5 |
| PlanCraft | 89 | 1,602 | 30 | 5e-6 |
| BabyAI | 25 | 450 | 20 | 5e-6 |

The main recipe draws 32 original and 32 hindsight demonstrations. Verified continuations are additional, up to four training draws when the bank has data. Physical padded batches and microbatches are in `configs/`. Evaluation and SFT costs are separate from online training costs.

`odm` is a retained internal trainer-controller label used by several historical modules. Selecting `--method dchl` installs `DecisionCoverageBuffer` and the DCHL trainer, not the external DynamixSFT baseline.

## 3. Evaluate the saved model

PlanCraft and BabyAI supply replica-managing launchers:

```bash
python scripts/run_plancraft_evaluation.py --model CHECKPOINT --run runs/eval_pc --gpus 0,1
python scripts/run_babyai_evaluation.py --model CHECKPOINT --run runs/eval_baby --gpus 0,1
```

Build the NVML helper as in installation before using these launchers. They terminate only their own spawned servers. Use the same prepared data and template as training. PlanCraft supports `--date-string` for its recorded template date.

ALFWorld uses the public evaluator plus the action-contract bridge. Start an OpenAI-compatible vLLM server with the saved model, then inspect and invoke:

```bash
export ALFWORLD_ACTION_CONTRACT=1 ALFWORLD_ETO_MAP_PUT=1
python scripts/evaluate_alfworld_eto_dialogue.py --help
```

Evaluate the full 140 seen and 134 unseen tasks with greedy decoding, 40 steps, and the same action contract. Report `(seen_successes + unseen_successes) / 274 × 100`, not the unweighted mean of the two split percentages. PlanCraft and BabyAI report 480- and 90-task success percentages respectively. Keep per-task IDs to verify complete coverage.

## 4. Resume after interruption

Use the original run directory with `--resume`. A usable checkpoint needs matching HF weights, optimizer and scheduler state for every rank, random states, and `online_state.json`. The tracker alone does not establish checkpoint completeness.

If logging advanced beyond the last saved update, archive those uncommitted metric rows and round directories outside `online/`, then retain the metric prefix through the checkpoint step. Resume from the following step. Preserve the interrupted runtime segment for total-cost accounting; the backend writes a new `runtime.json` for the resumed process. The release launcher rejects an unarchived metric tail.

Do not change the device world size, task schedule, template, method, loss normalization or common SFT while resuming. Model-only loading is a new training run, not recovery of the original online experiment.

## Validation scope

The release includes CPU behavior tests, Python compilation, shell syntax checks, checks of all 24 online launch plans, and source/asset scanning. These establish export integrity and local behavioral invariants. They do not establish a clean-machine reproduction of all training scores. Runtime assets and third-party package compatibility must be validated in the selected training environment before a full run.
