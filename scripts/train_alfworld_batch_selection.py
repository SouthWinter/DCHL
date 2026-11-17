"""Commit the better of two execution-valid batches under the current AdamW."""
import json
import time
import torch
import torch.distributed as dist
from tensordict.tensorclass import NonTensorData
from verl.utils import tensordict_utils as tu
from agent_probe_state import AdamProbeState
from alfworld_hsl_batch import InMemorySFTDataset, rank_records
from alfworld_batch_selection import SelectionBuffer, reference_records
from alfworld_continuation_compression import choose_jobs, evidence_entry
import train_alfworld_verified_recovery as parent


class SelectionTrainer(parent.RecoveryTrainer):
    collector_script = 'collect_alfworld_continuation_compression.py'

    def prepare_batch(self, rows, config, meta, update=True):
        local = rank_records(rows, dist.get_rank(), dist.get_world_size(), meta['micro_batch_size_per_gpu'])
        ds = InMemorySFTDataset(local, self.model_config.tokenizer, config)
        data = tu.get_tensordict(tensor_dict=self.collate_fn([ds[i] for i in range(len(ds))]),
                                non_tensor_dict=dict(meta, global_batch_size=len(ds)*dist.get_world_size(),
                                                     update_lr_scheduler=update, compute_loss=update))
        lengths = self._get_batch_seqlens(data)
        tu.assign_non_tensor(data, global_token_num=NonTensorData(lengths))
        return data, ds, lengths

    def score(self, data):
        output = self.training_client.infer_batch(data)
        total = torch.zeros((), dtype=torch.float64, device=torch.cuda.current_device())
        for logp, mask in zip(output['log_probs'].unbind(), data['loss_mask'].unbind()):
            shifted = torch.roll(mask, -1).clone()
            shifted[-1] = 0
            total -= (logp.float()*shifted.to(logp.device)).sum().double()
        dist.all_reduce(total)
        return float(total)

    def choose_batch_update(self, package, buffer, demos, config, meta, step, root, data_a, ds_a, lengths_a):
        rank = dist.get_rank()
        payload, proposal = [None], None
        if rank == 0:
            try:
                proposal = buffer.propose(package, step, int(self.config.trainer.seed))
                if proposal:
                    references, reference_info = reference_records(demos, package, proposal, step, int(self.config.trainer.seed))
                    payload[0] = dict(rows=proposal['rows'], provenance=proposal['provenance'],
                                      references=references, reference_info=reference_info)
            except Exception as exc:
                payload[0] = dict(error=f'{type(exc).__name__}: {exc}')
        dist.broadcast_object_list(payload, src=0)
        item = payload[0]
        if item and 'error' in item:
            raise RuntimeError(item['error'])
        if not item or not item['references']:
            output = self.training_client.train_batch(data=data_a)
            return output, ds_a, lengths_a, dict(selected='A', reason='No alternative or independent reference', updates_executed=1)
        started = time.monotonic()
        data_b, ds_b, lengths_b = self.prepare_batch(item['rows'], config, meta)
        reference, _, reference_lengths = self.prepare_batch(item['references'], config, meta, update=False)
        state = AdamProbeState(self.engine.module, self.engine.optimizer, self.engine.lr_scheduler)
        output_a = self.training_client.train_batch(data=data_a)
        score_a = self.score(reference)
        winner_a = AdamProbeState(self.engine.module, self.engine.optimizer, self.engine.lr_scheduler)
        state.restore()
        if step <= 2:
            state.assert_restored()
        output_b = self.training_client.train_batch(data=data_b)
        score_b = self.score(reference)
        choose_b = score_b < score_a
        if not choose_b:
            winner_a.restore()
            if step <= 2:
                winner_a.assert_restored()
        output, ds, lengths = (output_b, ds_b, lengths_b) if choose_b else (output_a, ds_a, lengths_a)
        if choose_b:
            package['rows'] = item['rows']
            package['provenance'] = item['provenance']
            if rank == 0:
                buffer.commit_proposal(proposal)
        report = dict(selected='B' if choose_b else 'A', score_a=score_a, score_b=score_b,
                      updates_executed=2, updates_committed=1, selection_seconds=time.monotonic()-started,
                      train_tokens_a=sum(lengths_a), train_tokens_b=sum(lengths_b),
                      reference_tokens=sum(reference_lengths)*2, reference=item['reference_info'],
                      action_rows_a=len(ds_a)*dist.get_world_size(), action_rows_b=len(ds_b)*dist.get_world_size())
        if rank == 0:
            (root/'rounds'/f'{step:05d}'/'batch_selection.json').write_text(json.dumps(report, indent=2)+'\n')
            (root/'rounds'/f'{step:05d}'/'alternative_provenance.json').write_text(json.dumps(item['provenance'], indent=2)+'\n')
        return output, ds, lengths, report


if __name__ == '__main__':
    parent.choose_jobs = choose_jobs
    parent.recovery_entry = evidence_entry
    online = parent.online
    online.HSLBuffer = SelectionBuffer
    online.mixed_records = parent.mixed_records
    online.ODMController = parent.BufferValueController
    online.OnlineHSLTrainer = SelectionTrainer
    online.main()
