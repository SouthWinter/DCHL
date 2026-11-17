"""Synchronous online HSL using the existing FSDP2 SFT engine.

Current actor -> resident vLLM -> real environment rollouts -> hindsight buffer
-> mixed trajectory loss -> one optimizer update. No PPO objective is involved.
"""
from pathlib import Path
from functools import partial
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request

import hydra
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from tensordict.tensorclass import NonTensorData

import verl.trainer.sft_trainer as sft
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group, destroy_global_process_group
from verl.utils.device import auto_set_device

from alfworld_hsl_batch import InMemorySFTDataset, group_demonstrations, mixed_records, rank_records, epoch_task_batch
from alfworld_hsl_buffer import HSLBuffer
from alfworld_hsl_loss import hsl_loss
from alfworld_hsl_credit import GoalSamplingController
from alfworld_odm import ODMController
from alfworld_hsl_weight_sync import sync_engine_weights


def json_value(value):
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


def tuple_tree(value):
    return tuple(map(tuple_tree, value)) if isinstance(value, (list, tuple)) else value


class OnlineHSLTrainer(sft.SFTTrainer):
    owned_service = None

    def start_owned_service(self, root):
        """Launch after actor initialization and the launcher's resource check."""
        if self.rank != 0 or not self.config.hsl.get('start_service', False):
            return
        from urllib.parse import urlparse
        endpoint = urlparse(self.config.hsl.endpoint)
        if endpoint.hostname != '127.0.0.1':
            raise ValueError('owned rollout service binds to localhost')
        env = {k: v for k, v in os.environ.items()
               if k not in {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                            'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE'}
               and not k.startswith('TORCHELASTIC_')}
        env.update(CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0],
                   VLLM_USE_V1='1', OMP_NUM_THREADS='2')
        if 'ALFWORLD_NVML_PHYSICAL_INDICES' in env:
            env['ALFWORLD_NVML_PHYSICAL_INDICES'] = env['ALFWORLD_NVML_PHYSICAL_INDICES'].split(',')[0]
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        # vLLM 0.8.5 V1 includes OTHER processes in its aggregate budget.
        # Add an explicit 14 GiB service allowance to current actor usage, rather
        # than letting the idle actor's small footprint inflate the KV cache.
        service_bytes = float(self.config.hsl.get('service_memory_gib', 14)) * 1024**3
        fraction = (total - free + service_bytes) / total
        if fraction >= 1:
            raise RuntimeError('insufficient GPU space for the resident rollout service')
        (root / 'rollout_memory_budget.json').write_text(json.dumps(dict(
            initial_used_bytes=total - free, service_allowance_bytes=service_bytes,
            vllm_aggregate_fraction=fraction)) + '\n')
        command = [sys.executable, '-u', str(Path(__file__).with_name('serve_alfworld_hsl.py')),
            '--model', self.config.model.path, '--served-model-name', 'qwen3-4b',
            '--port', str(endpoint.port), '--dtype', 'bfloat16', '--max-model-len', str(self.config.hsl.get('max_model_len', 8192)),
            '--max-num-seqs', str(int(self.config.hsl.get('service_max_num_seqs', 16))), '--gpu-memory-utilization',
            str(fraction),
            '--enforce-eager', '--disable-log-requests']
        if self.config.hsl.get('service_disable_prefix_cache', False):
            command.append('--no-enable-prefix-caching')
        if os.environ.get('ALFWORLD_FIXED_CHAT_TEMPLATE'):
            command.extend(['--chat-template', os.environ['ALFWORLD_FIXED_CHAT_TEMPLATE']])
        (root / 'rollout_service_config.json').write_text(json.dumps(dict(
            command=command, cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],
            max_num_seqs=int(self.config.hsl.get('service_max_num_seqs', 16)),
            disable_prefix_cache=bool(self.config.hsl.get('service_disable_prefix_cache', False)))) + '\n')
        self.service_log = (root / 'rollout_server.log').open('a')
        self.owned_service = subprocess.Popen(command, env=env, stdout=self.service_log,
            stderr=subprocess.STDOUT, start_new_session=True)
        (root / 'rollout_server.pid').write_text(str(self.owned_service.pid) + '\n')
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.owned_service.poll() is not None:
                raise RuntimeError('owned rollout service exited during initialization')
            try:
                with urllib.request.urlopen(self.config.hsl.endpoint + '/health', timeout=2):
                    return
            except (OSError, TimeoutError):
                time.sleep(2)
        raise TimeoutError('owned rollout service did not initialize within 600 seconds')

    def close_owned_service(self):
        if self.owned_service is not None:
            if self.owned_service.poll() is None:
                os.killpg(self.owned_service.pid, signal.SIGTERM)
                try:
                    self.owned_service.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(self.owned_service.pid, signal.SIGKILL)
                    self.owned_service.wait()
            self.service_log.close()

    def _build_engine(self):
        # SFTTrainer._build_engine installs its default loss after _build_config.
        # Register on the actual worker here, after that initialization.
        super()._build_engine()
        self.loss_fn = partial(hsl_loss, config=None)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)

    def collect_round(self, step, tasks, rng, version, root):
        """Rank-zero coordinator; six independent CPU environment workers."""
        directory = root / 'rounds' / f'{step:05d}'
        directory.mkdir(parents=True, exist_ok=True)
        from alfworld_data_schedule import round_tasks, completed_rows
        stable = bool(self.config.hsl.get('stable_data_schedule', False))
        selected = round_tasks(tasks, int(self.config.hsl.get('rollouts_per_step', 18)),
            int(self.config.trainer.seed), step, rng, stable)
        task_file = directory / 'tasks.jsonl'
        task_file.write_text(''.join(json.dumps(row) + '\n' for row in selected))
        output = directory / 'rollouts.jsonl'
        command = [str(self.config.hsl.environment_python), '-u',
            str(Path(__file__).with_name('collect_alfworld_value_calibration.py')),
            '--tasks', str(task_file), '--output', str(output),
            '--endpoint', self.config.hsl.endpoint,
            '--data-root', self.config.hsl.data_root,
            '--policy-version', version, '--hindsight', '--no-logprobs',
            '--relevance-rule', str(self.config.hsl.get('relevance_rule', 'backward')),
            '--num-shards', '1', '--shard-index', '0', '--repeats', '1',
            '--workers', str(self.config.hsl.get('workers', 6)), '--max-steps', '40',
            '--seed', str(int(self.config.trainer.seed) + step * 100003)]
        started = time.monotonic()
        if self.config.hsl.get('guidance_bank'):
            command += ['--guidance-bank',str(self.config.hsl.guidance_bank),
                        '--guided-count',str(self.config.hsl.get('guided_count',4)),
                        '--record-generation-usage']
        if self.config.hsl.get('correction_backend'):
            command += ['--correction-backend',str(self.config.hsl.correction_backend),
                        '--planner-timeout',str(self.config.hsl.get('planner_timeout',5)),
                        '--guided-count',str(self.config.hsl.get('guided_count',4)),
                        '--record-generation-usage']
        if self.config.hsl.get('transition_feedback', False):
            command += ['--record-generation-usage']
        with (directory / 'collector.log').open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='1'), check=True)
        # Last successful copy wins if the collector resumed an interrupted round.
        rows = completed_rows(map(json.loads, output.read_text().splitlines()), len(selected), stable)
        relabeled = [segment for r in rows for segment in r.get('hindsight_segments', [r['hindsight']])]
        metrics = dict(
            collection_seconds=time.monotonic() - started, rollout_count=len(rows),
            successes=sum(r['won'] for r in rows),
            relabel_worker_seconds=sum(h['wall_seconds'] for h in relabeled),
            relevance_worker_seconds=sum(h.get('relevance_worker_seconds', 0.) for h in relabeled),
            generated_goals=sum(len(h['goals']) for h in relabeled))
        if self.config.hsl.get('transition_feedback', False):
            metrics['transition_feedback'] = dict(
                prompt_tokens=sum(u['prompt_tokens'] for r in rows for u in r['generation_usage']),
                completion_tokens=sum(u['completion_tokens'] for r in rows for u in r['generation_usage']),
                extra_teacher_calls=0, extra_environment_episodes=0)
        if self.config.hsl.get('guidance_bank'):
            metrics['inline_guidance'] = dict(
                selected=sum(r['guidance']['selected'] for r in rows),
                activated=sum(r['guidance']['activated'] for r in rows),
                guided_actions=sum(r['guidance']['guided_actions'] for r in rows),
                original_goal_successes=sum(r['won'] for r in rows if r['guidance']['selected']),
                prompt_tokens=sum(u['prompt_tokens'] for r in rows for u in r['generation_usage']),
                completion_tokens=sum(u['completion_tokens'] for r in rows for u in r['generation_usage']))
        if self.config.hsl.get('correction_backend') == 'goal_operation_label':
            probes = [r['action_labeling']['probe'] for r in rows if r['action_labeling']['probe']]
            solvers = [p[k] for p in probes for k in ('teacher_solver','student_after_solver') if k in p]
            metrics['goal_operation_correction'] = dict(
                selected=sum(r['action_labeling']['selected'] for r in rows),
                probed_states=len(probes),planner_calls=len(probes)+sum(p.get('student_query_called',False) for p in probes),
                accepted_labels=sum(r['action_labeling']['accepted_labels'] for r in rows),
                candidate_branch_attempts=len(probes),
                completed_candidate_branch_actions=sum(p['teacher_solver'].get('expert_executed_actions',0) for p in probes),
                candidate_branch_execution_unknown=sum('expert_executed_actions' not in p['teacher_solver'] for p in probes),
                solver_worker_seconds=sum(s['seconds'] for s in solvers),extra_environment_steps=0,
                teacher_model_calls=0,prompt_tokens=sum(u['prompt_tokens'] for r in rows for u in r['generation_usage']),
                completion_tokens=sum(u['completion_tokens'] for r in rows for u in r['generation_usage']))
        elif self.config.hsl.get('correction_backend') == 'cost_screened_label':
            probes = [r['action_labeling']['probe'] for r in rows if r['action_labeling']['probe']]
            solvers = [p[k] for p in probes for k in ('before_solver', 'after_solver') if k in p]
            metrics['cost_screened_labels'] = dict(
                selected=sum(r['action_labeling']['selected'] for r in rows),
                probed_states=len(probes), planner_calls=len(probes) + sum(p.get('after_query_called', False) for p in probes),
                accepted_labels=sum(r['action_labeling']['accepted_labels'] for r in rows),
                solver_worker_seconds=sum(s['seconds'] for s in solvers),
                teacher_actions=0, extra_environment_steps=0,
                prompt_tokens=sum(u['prompt_tokens'] for r in rows for u in r['generation_usage']),
                completion_tokens=sum(u['completion_tokens'] for r in rows for u in r['generation_usage']))
        elif self.config.hsl.get('correction_backend') == 'equivalent_condition':
            metrics['condition_transfer'] = dict(
                selected=sum(r['condition_transfer']['selected'] for r in rows),
                activated=sum(r['condition_transfer']['activated'] for r in rows),
                original_goal_successes=sum(r['won'] for r in rows if r['condition_transfer']['selected']),
                bound_goals=sum(r['condition_transfer']['bound_goals'] for r in rows),
                bound_supervised_actions=sum(g['relevant_actions'] for r in rows for g in r['hindsight']['goals'] if g.get('original_request_bound')),
                teacher_actions=0, solver_worker_seconds=0,
                prompt_tokens=sum(u['prompt_tokens'] for r in rows for u in r['generation_usage']),
                completion_tokens=sum(u['completion_tokens'] for r in rows for u in r['generation_usage']))
        elif self.config.hsl.get('correction_backend') == 'cycle_resample':
            metrics['cycle_resample'] = dict(
                **{k: sum(r['cycle_resample'][k] for r in rows) for k in (
                    'triggered','removed_actions','physical_attempts','resets','neural_actions',
                    'prefix_goals','continuation_goals','omitted_shared_goals','teacher_calls','planner_calls')},
                prompt_tokens=sum(u['prompt_tokens'] for r in rows for u in r['generation_usage']),
                completion_tokens=sum(u['completion_tokens'] for r in rows for u in r['generation_usage']))
        elif self.config.hsl.get('correction_backend') == 'budget_restart':
            metrics['budget_restart'] = dict(
                triggered=sum(r['budget_restart']['triggered'] for r in rows),
                physical_attempts=sum(r['budget_restart']['physical_attempts'] for r in rows),
                neural_actions=sum(r['steps'] for r in rows),
                resets=sum(r['budget_restart']['resets'] for r in rows),
                first_prefix_goals=sum(r['budget_restart']['first_prefix_goals'] for r in rows),
                fresh_goals=sum(r['budget_restart']['fresh_goals'] for r in rows),
                teacher_calls=0, planner_calls=0,
                prompt_tokens=sum(u['prompt_tokens'] for r in rows for u in r['generation_usage']),
                completion_tokens=sum(u['completion_tokens'] for r in rows for u in r['generation_usage']))
        elif self.config.hsl.get('correction_backend'):
            solvers = [r['correction']['solver'] for r in rows if r['correction']['activated']]
            metrics['inline_correction'] = dict(
                selected=sum(r['correction']['selected'] for r in rows), activated=len(solvers),
                accepted=sum(s['accepted'] for s in solvers),
                teacher_actions=sum(s['teacher_actions'] for s in solvers),
                solver_worker_seconds=sum(s['seconds'] for s in solvers),
                original_goal_successes=sum(r['won'] for r in rows if r['correction']['selected']),
                prompt_tokens=sum(u['prompt_tokens'] for r in rows for u in r['generation_usage']),
                completion_tokens=sum(u['completion_tokens'] for r in rows for u in r['generation_usage']))
            if self.config.hsl.correction_backend == 'public_grounded_planner':
                metrics['public_correction'] = {
                    key: sum(r['public_correction'].get(key, 0) for r in rows)
                    for key in ('bound_goals', 'original_correction_goals',
                                'original_correction_prefix_masked',
                                'expert_targets_without_public_witnesses_masked',
                                'kept_action_targets', 'expert_actions',
                                'zero_target_goals_excluded')}
            if self.config.hsl.correction_backend == 'public_acquisition_planner':
                metrics['acquisition_correction'] = {
                    key: sum(r['acquisition_correction'][key] for r in rows)
                    for key in ('bound_goals', 'planner_called', 'public_plan_accepted')}
        return relabeled, metrics

    def fit(self):
        # HF weights are the sole model copy in this run's resumable saves.
        # Export them at every scheduled save, including intermediate saves.
        os.environ.pop('VES_HF_EXPORT_STEP', None)
        root = Path(self.config.hsl.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        rank, world = dist.get_rank(), dist.get_world_size()
        startup = [None]
        if rank == 0:
            try:
                self.start_owned_service(root)
            except Exception as error:
                startup[0] = str(error)
        dist.broadcast_object_list(startup, src=0)
        if startup[0]:
            raise RuntimeError(startup[0])
        microbatch = int(self.config.data.micro_batch_size_per_gpu)
        start = self.resume_global_step
        if start and hasattr(self.engine, 'checkpoint_manager'):
            # Track the restored checkpoint for save-after-success rotation.
            # The framework only initializes this list for its overwrite mode.
            self.engine.checkpoint_manager.previous_saved_paths = [
                str(Path(self.config.trainer.default_local_dir) / f'global_step_{start}')]
        state = None
        controller = None
        demos = buffer = None
        sampler = self.config.hsl.get('sampler', 'behavior' if self.config.hsl.get('adaptive_sampling', False) else 'base')
        paired_feedback = bool(self.config.hsl.get('paired_feedback', False))
        stable_data_schedule = bool(self.config.hsl.get('stable_data_schedule', False))
        if paired_feedback and sampler != 'behavior':
            raise ValueError('Paired feedback requires the behavior sampler')
        if sampler not in ('base', 'behavior', 'odm'):
            raise ValueError(f'Unknown sampler: {sampler}')
        if sampler == 'odm' and int(self.config.hsl.get('batch_trajectories', 32)) != 32:
            raise ValueError('ODM reward statistics currently require 32 trajectory draws')
        if rank == 0:
            if sampler == 'behavior':
                controller = GoalSamplingController(float(self.config.hsl.get('sampler_learning_rate', 32 ** -.5)))
            elif sampler == 'odm':
                controller = ODMController(float(self.config.hsl.get('odm_alpha', .9)), 1)
            demos = group_demonstrations(pq.read_table(self.config.data.train_files).to_pylist())
            tasks = [json.loads(line) for line in Path(self.config.hsl.task_pool).read_text().splitlines()]
            rng = random.Random(int(self.config.trainer.seed))
            buffer = HSLBuffer(capacity=int(self.config.hsl.get('buffer_capacity', 100)),
                               seed=int(self.config.trainer.seed))
            if start:
                state_path = Path(self.config.trainer.default_local_dir) / f'global_step_{start}' / 'online_state.json'
                state = json.loads(state_path.read_text())
                from alfworld_data_schedule import check_resume_schedule
                check_resume_schedule(state, stable_data_schedule)
                if state['step'] != start:
                    raise ValueError('online state and optimizer checkpoint steps differ')
                if state.get('sampler', sampler) != sampler:
                    raise ValueError('Cannot resume a different sampler')
                if state.get('paired_feedback', False) != paired_feedback:
                    raise ValueError('Cannot change feedback protocol on resume')
                rng.setstate(tuple_tree(state['data_rng']))
                buffer = HSLBuffer.from_state_dict(state['buffer'])
                if controller is not None:
                    controller = (ODMController if sampler == 'odm' else GoalSamplingController).from_state_dict(state['sampling_controller'])
            buffer.stable_data_schedule = stable_data_schedule
        data_config = OmegaConf.to_container(self.config.data, resolve=True)
        data_config['loss_weight_key'] = 'trajectory_action_coefficient'
        data_config['sampling_statistics'] = sampler == 'odm'
        meta = dict(use_remove_padding=bool(self.config.model.use_remove_padding), use_dynamic_bsz=False,
            max_token_len_per_gpu=self.config.data.max_token_len_per_gpu,
            micro_batch_size_per_gpu=microbatch, temperature=1.0,
            pad_mode=self.config.data.pad_mode, pad_token_id=self.model_config.tokenizer.pad_token_id)

        for step in range(start + 1, self.total_training_steps + 1):
            # Apply throughput changes only at an optimizer-update boundary.
            # All original records/weights are retained; only zero-loss padding
            # and forward/backward partitioning change.
            requested_microbatch = [microbatch]
            control = root.parent / 'microbatch.json'
            if rank == 0 and control.exists():
                requested_microbatch[0] = int(json.loads(control.read_text())['microbatch_per_gpu'])
            dist.broadcast_object_list(requested_microbatch, src=0)
            microbatch = requested_microbatch[0]
            if microbatch < 1:
                raise ValueError('microbatch_per_gpu must be positive')
            meta['micro_batch_size_per_gpu'] = microbatch
            version = f'{root.name}:step-{step - 1}'
            synced = sync_engine_weights(self.engine, self.config.hsl.endpoint, version)
            package = [None]
            if rank == 0:
                try:
                    relabeled, collection_metrics = self.collect_round(step, tasks, rng, version, root)
                    reward = collection_metrics['successes'] / collection_metrics['rollout_count']
                    if hasattr(buffer, 'observe_collection'):
                        buffer.observe_collection(step, reward)
                    feedback = controller.observe(step, reward) if sampler == 'behavior' and not paired_feedback else None
                    buffer.append_round(relabeled, version)
                    if hasattr(self, 'prepare_round_training'):
                        self.prepare_round_training(buffer, demos, data_config, step, root)
                    goal_logits = controller.logits(buffer, step) if sampler == 'odm' else (controller.logits if controller is not None else None)
                    from alfworld_data_schedule import row_rng
                    rows, provenance = mixed_records(demos, buffer,
                        row_rng(int(self.config.trainer.seed), step, rng, stable_data_schedule),
                        trajectories=int(self.config.hsl.get('batch_trajectories', 32)),
                        hindsight_fraction=float(self.config.hsl.get('hindsight_fraction', .3)),
                        goal_logits=goal_logits,
                        original_tasks=epoch_task_batch(demos, step,
                            int(self.config.hsl.get('batch_trajectories', 32)), int(self.config.trainer.seed)))
                    package[0] = dict(rows=rows, provenance=provenance)
                    if hasattr(self, 'transform_mixed_batch'):
                        package[0]['rows'], package[0]['prefix_coalescing'] = self.transform_mixed_batch(
                            rows, provenance, data_config, step, root)
                    sampling_info = None
                    if sampler == 'behavior':
                        sampling_info = dict(feedback=feedback, sampling=controller.remember(
                            step, reward, provenance, buffer.last_sample_distribution))
                except Exception as error:
                    package[0] = {'error': f'{type(error).__name__}: {error}'}
            dist.broadcast_object_list(package, src=0)
            if 'error' in package[0]:
                raise RuntimeError(package[0]['error'])
            local_rows = rank_records(package[0]['rows'], rank, world, microbatch)
            preparation_metrics = None
            if hasattr(self, 'prepare_reference_transfer'):
                local_rows, preparation_metrics = self.prepare_reference_transfer(local_rows, data_config, meta, step, root)
                if rank == 0 and hasattr(buffer, 'observe_preparation'):
                    buffer.observe_preparation(preparation_metrics)
            dataset = InMemorySFTDataset(local_rows, self.model_config.tokenizer, data_config)
            batch = self.collate_fn([dataset[i] for i in range(len(dataset))])
            data = tu.get_tensordict(tensor_dict=batch,
                non_tensor_dict=dict(meta, global_batch_size=len(dataset) * world))
            lengths = self._get_batch_seqlens(data)
            tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=NonTensorData(lengths))
            started = time.monotonic()
            selection_report = None
            if hasattr(self, 'choose_batch_update'):
                output, dataset, lengths, selection_report = self.choose_batch_update(
                    package[0], buffer, demos, data_config, meta, step, root, data, dataset, lengths)
                if rank == 0:
                    provenance = package[0]['provenance']
            else:
                output = self.training_client.train_batch(data=data)
            update_seconds = time.monotonic() - started
            paired_metrics = None
            if paired_feedback:
                from alfworld_hsl_paired_feedback import collect_post_update
                post_version = f'{root.name}:step-{step}'
                post_sync = sync_engine_weights(self.engine, self.config.hsl.endpoint, post_version)
                post_error = [None]
                if rank == 0:
                    try:
                        paired_metrics = collect_post_update(self.config, step, post_version, root)
                        post_reward = paired_metrics['after_successes'] / paired_metrics['rollout_count']
                        sampling_info['feedback'] = controller.observe(step + 1, post_reward)
                        sampling_info['feedback'].update(feedback_mode='same_tasks_post_update',
                                                         feedback_collection_step=step)
                        paired_metrics['sync'] = post_sync
                    except Exception as error:
                        post_error[0] = f'{type(error).__name__}: {error}'
                dist.broadcast_object_list(post_error, src=0)
                if post_error[0]:
                    raise RuntimeError(post_error[0])
            if rank == 0:
                if sampler == 'odm':
                    sampling_info = controller.observe(step, provenance, buffer.last_sample_distribution, tu.get(output, 'metrics'))
                metrics = dict(step=step, rollout_policy_version=version,
                    actor_policy_version=f'{root.name}:step-{step}',
                    **collection_metrics, **synced,
                    update_seconds=update_seconds, microbatch_per_gpu=microbatch,
                    action_records=len(package[0]['rows']),
                    padded_action_records=len(dataset) * world, input_tokens=sum(lengths),
                    training_metrics=tu.get(output, 'metrics'))
                if 'prefix_coalescing' in package[0]:
                    metrics['prefix_coalescing'] = package[0]['prefix_coalescing']
                if sampling_info is not None:
                    metrics['sampling_controller'] = sampling_info
                if hasattr(buffer, 'observe_training'):
                    metrics[getattr(buffer, 'feedback_metric_key', 'soft_rehearsal')] = buffer.observe_training(tu.get(output, 'metrics'))
                if paired_metrics is not None:
                    metrics['paired_feedback'] = paired_metrics
                if preparation_metrics is not None:
                    metrics[getattr(self, 'preparation_metric_key', 'reference_transfer')] = preparation_metrics
                if selection_report is not None:
                    metrics['batch_selection'] = selection_report
                with (root / 'metrics.jsonl').open('a') as log:
                    log.write(json.dumps(metrics, default=json_value) + '\n')
                (root / 'rounds' / f'{step:05d}' / 'batch_provenance.json').write_text(
                    json.dumps(package[0]['provenance'], indent=2) + '\n')
                print(json.dumps(metrics, default=json_value), flush=True)
            if step % int(self.save_freq) == 0 or step == self.total_training_steps:
                self.ckpt_handler.save_checkpoint(step=step)
                if rank == 0:
                    path = Path(self.config.trainer.default_local_dir) / f'global_step_{step}' / 'online_state.json'
                    temporary = path.with_suffix('.tmp')
                    saved = dict(step=step, buffer=buffer.state_dict(),
                        data_rng=rng.getstate(), sampler=sampler, paired_feedback=paired_feedback,
                        stable_data_schedule=stable_data_schedule, sampling_controller=(
                            controller.state_dict() if controller is not None else None))
                    from alfworld_data_schedule import frozen_schedule_state
                    frozen_schedule = frozen_schedule_state()
                    if frozen_schedule is not None:
                        saved['frozen_task_schedule'] = frozen_schedule
                    if hasattr(self, 'checkpoint_extra_state'):
                        saved['trainer_extra_state'] = self.checkpoint_extra_state()
                    temporary.write_text(json.dumps(saved) + '\n')
                    temporary.replace(path)
                dist.barrier()
        if rank == 0:
            (root / 'ONLINE_COMPLETE.json').write_text(json.dumps(dict(
                steps=self.total_training_steps, kind=sampler)) + '\n')


@hydra.main(config_path=str(Path(sft.__file__).parent / 'config'),
            config_name='sft_trainer_engine', version_base=None)
def main(config):
    os.environ['NO_PROXY'] = '127.0.0.1,localhost'
    os.environ['no_proxy'] = '127.0.0.1,localhost'
    auto_set_device(config)
    if 'model' not in config.checkpoint.load_contents and config.trainer.resume_mode != 'disable':
        # The FP32 HF export is the single model copy in a resumable checkpoint.
        # Load those weights before restoring its optimizer, scheduler and RNG.
        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
        resume = config.trainer.get('resume_from_path')
        if resume is None and Path(config.trainer.default_local_dir).exists():
            resume = find_latest_ckpt_path(config.trainer.default_local_dir)
        if resume:
            model = Path(resume) / 'huggingface'
            if not (model / 'config.json').is_file():
                raise ValueError('resume checkpoint is missing its full model export')
            config.model.path = str(model)
    initialize_global_process_group()
    trainer = None
    try:
        trainer = OnlineHSLTrainer(config)
        trainer.fit()
    finally:
        if trainer is not None:
            trainer.close_owned_service()
        destroy_global_process_group()


if __name__ == '__main__':
    main()
