#!/usr/bin/env bash
set -euo pipefail

data_file=${1:?data parquet required}
target_step=${2:?target global step required}
run_root=${3:?run root required}
checkpoint_root=${4:?checkpoint root required}
resume_mode=${5:?resume mode required: disable or auto}
loss_weight_key=${6:-}
gpu_list=${7:-4,5,6,7}

source_project=${DCHL_ROOT:?source env.sh first}
verl_root=${DCHL_VERL_ROOT:-$source_project/external/verl}
python_bin=${DCHL_PYTHON:-python}
model=${ALFWORLD_INITIAL_MODEL_PATH:-${DCHL_BASE_MODEL:?set DCHL_BASE_MODEL}}
data_shuffle=${ALFWORLD_DATA_SHUFFLE:-true}
save_freq=${ALFWORLD_SAVE_FREQ:-$target_step}
IFS=',' read -r -a gpus <<<"$gpu_list"
gpu_count=${#gpus[@]}
(( gpu_count >= 1 && gpu_count <= 8 )) || { echo "one to eight GPUs are required" >&2; exit 2; }
[[ "$resume_mode" == disable || "$resume_mode" == auto ]] || { echo "invalid resume mode" >&2; exit 3; }
[[ "$data_shuffle" == true || "$data_shuffle" == false ]] || { echo "invalid data shuffle flag" >&2; exit 3; }
[[ "$save_freq" =~ ^[1-9][0-9]*$ ]] || { echo "save frequency must be positive" >&2; exit 3; }
[[ -f "$data_file" && -f "$model/config.json" ]] || { echo "data or model is incomplete" >&2; exit 4; }
[[ ! -e "$run_root/COMPLETE" ]] || { echo "stage already complete: $run_root" >&2; exit 5; }

for gpu in "${gpus[@]}"; do
  if [[ ${ALFWORLD_TRAIN_SHARED_GPU:-0} == 1 ]]; then
    # Opt-in sharing authorized for this run; never stop other GPU processes.
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    ((free_mib >= ${ALFWORLD_TRAIN_MIN_FREE_MIB:-45056})) || { echo "GPU $gpu has insufficient free memory: $free_mib MiB" >&2; exit 6; }
    echo "Shared training GPU $gpu: $free_mib MiB free"
  else
    busy=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')
    [[ -z "$busy" ]] || { echo "GPU $gpu is busy: $busy" >&2; exit 6; }
  fi
done
mkdir -p "$checkpoint_root"
available_kib=$(df -Pk "$checkpoint_root" | awk 'NR==2 {print $4}')
min_free_gib=${ALFWORLD_MIN_FREE_GIB:-80}
(( available_kib >= min_free_gib * 1024 * 1024 )) || { echo "at least $min_free_gib GiB free space is required" >&2; exit 7; }

mkdir -p "$run_root/logs"
loss_weight_args=()
if [[ -n "$loss_weight_key" ]]; then
  loss_weight_args+=(data.loss_weight_key="$loss_weight_key")
fi

# CUDA ordinals can compact after a GPU drops off the bus; NVML indices do not.
# UUIDs bind the selected physical devices in both healthy and degraded nodes.
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(for gpu in "${gpus[@]}"; do
  nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader
done | paste -sd, -)
printf '%s\n' "$CUDA_VISIBLE_DEVICES" >"$run_root/logs/gpu_bindings.txt"
export PYTHONPATH=$source_project/scripts:$verl_root${PYTHONPATH:+:$PYTHONPATH}
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300
export VERL_SKIP_PRE_CHECKPOINT_EMPTY_CACHE=1
export VES_HF_EXPORT_STEP=$target_step

# Explicit opt-in for reproducibility diagnostics/new protocols. Historical
# launchers retain their recorded behavior until a new protocol selects this.
determinism_args=()
if [[ ${ALFWORLD_DETERMINISTIC_TRAINING:-0} == 1 ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:16:8
  export FLASH_ATTENTION_DETERMINISTIC=1
  determinism_args+=(engine.full_determinism=true 'engine.seed=${trainer.seed}')
fi

training_env=()
if [[ ${ALFWORLD_FILTER_NVML:-0} == 1 ]]; then
  nvml_shim=$(dirname "$0")/../cache/nvml_selected_devices.so
  [[ -f "$nvml_shim" ]] || { echo "Build scripts/nvml_selected_devices.c first" >&2; exit 8; }
  training_env+=("ALFWORLD_NVML_PHYSICAL_INDICES=$gpu_list"
    "LD_PRELOAD=$nvml_shim${LD_PRELOAD:+:$LD_PRELOAD}")
fi
started=$(date +%s)
set +e
/usr/bin/time -v -o "$run_root/logs/process_resources.txt" \
  ionice -c2 -n7 nice -n 5 env "${training_env[@]}" "$python_bin" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="$gpu_count" \
  -m "${ALFWORLD_SFT_MODULE:-verl.trainer.sft_trainer}" \
  data.train_files="$data_file" \
  data.val_files=null \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  "${loss_weight_args[@]}" \
  data.train_batch_size=64 \
  data.micro_batch_size_per_gpu="${ALFWORLD_TRAIN_MICRO_BATCH_SIZE:-4}" \
  data.max_token_len_per_gpu=2048 \
  data.use_dynamic_bsz=false \
  data.pad_mode=no_padding \
  data.max_length=2048 \
  data.truncation=error \
  data.num_workers=0 \
  data.shuffle="$data_shuffle" \
  data.ignore_input_ids_mismatch=true \
  model.path="$model" \
  model.lora_rank=0 \
  +model.enable_tied_meta_init=true \
  model.enable_gradient_checkpointing=true \
  model.use_remove_padding="${ALFWORLD_MODEL_REMOVE_PADDING:-true}" \
  +model.override_config.attn_implementation=sdpa \
  engine=fsdp \
  engine.strategy=fsdp2 \
  engine.fsdp_size="$gpu_count" \
  engine.param_offload=false \
  engine.optimizer_offload=false \
  engine.offload_policy=false \
  engine.dtype=bfloat16 \
  engine.model_dtype=fp32 \
  engine.use_torch_compile=false \
  optim.lr="${ALFWORLD_SFT_LR:-5e-6}" \
  optim.lr_scheduler_type=constant \
  optim.weight_decay=0.01 \
  optim.clip_grad=1.0 \
  checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  checkpoint.load_contents='[model,optimizer,extra]' \
  trainer.default_local_dir="$checkpoint_root" \
  trainer.max_ckpt_to_keep=1 \
  trainer.project_name=alfworld_hierarchical_data_allocation \
  trainer.experiment_name="alfworld_hierarchical_step_${target_step}" \
  trainer.logger=console \
  trainer.seed=20260822 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$target_step" \
  trainer.save_freq="$save_freq" \
  trainer.test_freq=-1 \
  trainer.resume_mode="$resume_mode" \
  trainer.n_gpus_per_node="$gpu_count" \
  trainer.nnodes=1 "${@:8}" "${determinism_args[@]}" \
  >"$run_root/logs/train.log" 2>&1
status=$?
set -e
ended=$(date +%s)

checkpoint=$checkpoint_root/global_step_$target_step
hf_model=$checkpoint/huggingface
if [[ $status -eq 0 && ! -f "$hf_model/config.json" ]]; then
  status=20
fi
printf '{"target_step":%s,"wall_seconds":%s,"exit_status":%s,"gpu_indices":[%s],"data":"%s","loss_weight_key":"%s","checkpoint":"%s"}\n' \
  "$target_step" "$((ended-started))" "$status" "$gpu_list" "$data_file" "$loss_weight_key" "$checkpoint" \
  >"$run_root/runtime.json"
if [[ $status -ne 0 ]]; then
  echo FAILED >"$run_root/status.txt"
  exit "$status"
fi
echo COMPLETE >"$run_root/status.txt"
touch "$run_root/COMPLETE"
