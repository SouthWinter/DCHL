"""Adapt Data Agent's Beta-PPO sample policy to masked action-token evidence."""
import json
import math

import torch
import torch.distributed as dist
from verl.utils import tensordict_utils as tu
from verl.utils import torch_functional as verl_F


def _chunked_entropy(logits):
    """Exact token entropy with bounded temporary vocabulary tensors."""
    flat = logits.reshape(-1, logits.shape[-1])
    entropy = torch.empty(flat.shape[0], device=logits.device, dtype=torch.float32)
    for start in range(0, flat.shape[0], 64):
        block = flat[start:start + 64].float()
        probabilities = torch.softmax(block, dim=-1)
        entropy[start:start + len(block)] = (
            torch.logsumexp(block, dim=-1) - (probabilities * block).sum(dim=-1))
    return entropy.reshape(logits.shape[:-1])


# The installed FSDP no-padding inference path calls this module function for
# entropy even when its engine flag requests chunking. Scope the replacement to
# Data Agent entrypoints, whose 64-candidate scoring requires entropy.
verl_F.entropy_from_logits = _chunked_entropy


class DataAgentMixin:
    def score_candidates(self, data, dataset, expected):
        tu.assign_non_tensor(data, calculate_entropy=True)
        output = self.training_client.infer_batch(data)
        local = []
        for row, logp, entropy, mask in zip(dataset.rows, output['log_probs'].unbind(),
                                              output['entropy'].unbind(), data['loss_mask'].unbind()):
            shifted = torch.roll(mask, -1).clone()
            shifted[-1] = 0
            values = -logp.float()[shifted.to(logp.device).bool()]
            entropies = entropy.float()[shifted.to(entropy.device).bool()]
            if values.numel() == 0:
                continue
            local.append(dict(index=row['data_agent_candidate_idx'],
                nll=float(values.mean()), nll_std=float(values.std(unbiased=False)),
                entropy=float(entropies.mean()), entropy_std=float(entropies.std(unbiased=False)),
                count=int(values.numel())))
        all_rows = [None] * dist.get_world_size()
        dist.all_gather_object(all_rows, local)
        if dist.get_rank() != 0:
            return None
        by_index = {}
        for shard in all_rows:
            for record in shard:
                index = record['index']
                if index in by_index:
                    raise ValueError('Duplicate scored candidate')
                by_index[index] = record
        if set(by_index) != set(range(expected)):
            raise ValueError(f'Candidate score coverage {len(by_index)}/{expected}')
        ordered = [by_index[i] for i in range(expected)]
        assert all(math.isfinite(value) for m in ordered for key,value in m.items() if key not in ('index','count'))
        return ordered

    def choose_batch_update(self, package, buffer, demos, config, meta, step,
                            root, data_a, ds_a, lengths_a):
        payload = [None]
        entries = None
        if dist.get_rank() == 0:
            try:
                entries = buffer.candidates(step, int(self.config.trainer.seed))
                payload[0] = dict(rows=buffer.scoring_rows(entries), count=len(entries)) if entries else None
            except Exception as exc:
                payload[0] = dict(error=f'{type(exc).__name__}: {exc}')
        dist.broadcast_object_list(payload, src=0)
        item = payload[0]
        if item and 'error' in item:
            raise RuntimeError(item['error'])
        proposal = None
        if item:
            # Full-sequence logits remain materialized; score one candidate at
            # a time while entropy uses bounded sequence chunks.
            # while retaining the original microbatch for the model update.
            score_meta = dict(meta, micro_batch_size_per_gpu=1)
            scores_data, scores_dataset, _ = self.prepare_batch(item['rows'], config, score_meta, update=False)
            candidate_metrics = self.score_candidates(scores_data, scores_dataset, item['count'])
            proposed = [None]
            if dist.get_rank() == 0:
                try:
                    states = [buffer.state_from_scores(m['nll'], m['entropy'],
                        m['nll_std'], m['entropy_std'], m['count']) for m in candidate_metrics]
                    proposal = buffer.select(entries, states, candidate_metrics, package,
                        step, int(self.config.trainer.seed), self.total_training_steps)
                    proposed[0] = dict(rows=proposal['rows'],provenance=proposal['provenance'])
                except Exception as exc:
                    proposed[0] = dict(error=f'{type(exc).__name__}: {exc}')
            dist.broadcast_object_list(proposed,src=0)
            if 'error' in proposed[0]:
                raise RuntimeError(proposed[0]['error'])
            data, dataset, lengths = self.prepare_batch(proposed[0]['rows'],config,meta)
            output = self.training_client.train_batch(data=data)
            package.update(proposed[0])
            if dist.get_rank() == 0:
                buffer.commit_proposal(proposal)
                (root/'rounds'/f'{step:05d}'/'data_agent_selection.json').write_text(
                    json.dumps(buffer.last_data_agent,indent=2)+'\n')
            report = dict(method='Data Agent Beta-PPO top-half adaptation',
                candidate_count=item['count'],selected=32,scored_action_rows=len(item['rows']),
                policy_mode=buffer.last_data_agent.get('mode') if dist.get_rank()==0 else None,
                ppo=buffer.last_data_agent.get('ppo') if dist.get_rank()==0 else None)
            return output,dataset,lengths,report
        output=self.training_client.train_batch(data=data_a)
        return output,ds_a,lengths_a,dict(method='Data Agent',candidate_count=0,selected=0)
