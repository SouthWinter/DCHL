"""Online SFT with DynamixSFT task-family arm probabilities and delta-loss reward."""
import json

import torch
import torch.distributed as dist

from alfworld_dynamixsft import DynamixSFTBuffer
from train_alfworld_direct_candidate import DirectCandidateTrainer, parent, choose_jobs, evidence_entry


class DynamixSFTTrainer(DirectCandidateTrainer):
    probe_interval = 10

    def group_losses(self, data, dataset, arms):
        output = self.training_client.infer_batch(data)
        totals = torch.zeros((len(arms), 2), dtype=torch.float64,
                             device=torch.cuda.current_device())
        indices = {arm: i for i, arm in enumerate(arms)}
        for row, logp, mask in zip(dataset.rows, output['log_probs'].unbind(),
                                   data['loss_mask'].unbind()):
            shifted = torch.roll(mask, -1).clone()
            shifted[-1] = 0
            count = shifted.sum().double()
            if count:
                index = indices[row['dynamix_arm']]
                totals[index, 0] -= (logp.float() * shifted.to(logp.device)).sum().double()
                totals[index, 1] += count
        dist.all_reduce(totals)
        if (totals[:, 1] <= 0).any():
            raise ValueError('Missing supervised probe tokens in an arm')
        return {arm: float(totals[i, 0] / totals[i, 1]) for i, arm in enumerate(arms)}

    def choose_batch_update(self, package, buffer, demos, config, meta, step,
                            root, data_a, ds_a, lengths_a):
        probing = step % self.probe_interval == 0
        payload = [None]
        if probing and dist.get_rank() == 0:
            try:
                rows, arms = buffer.probe_rows(step, int(self.config.trainer.seed))
                payload[0] = dict(rows=rows, arms=arms)
            except Exception as exc:
                payload[0] = dict(error=f'{type(exc).__name__}: {exc}')
        if probing:
            dist.broadcast_object_list(payload, src=0)
            if 'error' in payload[0]:
                raise RuntimeError(payload[0]['error'])
            probe_data, probe_dataset, _ = self.prepare_batch(
                payload[0]['rows'], config, meta, update=False)
            before = self.group_losses(probe_data, probe_dataset, payload[0]['arms'])
        output, dataset, lengths, report = super().choose_batch_update(
            package, buffer, demos, config, meta, step, root, data_a, ds_a, lengths_a)
        report.update(method='DynamixSFT task-family adaptation',
                      arm_probabilities=buffer.last_dynamix.get('arm_probabilities')
                      if dist.get_rank() == 0 else None,
                      reward_probe=probing)
        if probing:
            after = self.group_losses(probe_data, probe_dataset, payload[0]['arms'])
            if dist.get_rank() == 0:
                report['lookahead'] = buffer.observe_probe(before, after)
                (root/'rounds'/f'{step:05d}'/'dynamixsft_reward.json').write_text(
                    json.dumps(report['lookahead'], indent=2) + '\n')
        return output, dataset, lengths, report


if __name__ == '__main__':
    parent.choose_jobs = choose_jobs
    parent.recovery_entry = evidence_entry
    online = parent.online
    online.HSLBuffer = DynamixSFTBuffer
    online.mixed_records = parent.mixed_records
    online.ODMController = parent.BufferValueController
    online.OnlineHSLTrainer = DynamixSFTTrainer
    online.main()
