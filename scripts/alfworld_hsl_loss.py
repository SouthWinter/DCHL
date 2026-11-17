"""Trajectory-normalized loss for a preweighted HSL mixed batch.

Each action record's loss mask contains its complete scalar coefficient:
source mixture / sampled trajectories / trajectory action count. The HSL goal
weight enters through weighted trajectory sampling, so it is not applied again.
The coefficient multiplies the action's SUM token negative log probability.
No division by the (weighted) token count is performed.
"""
import torch


def hsl_loss(config, model_output, data, dp_group=None, feedback_slots=32):
    log_probs = model_output['log_probs']
    mask = data['loss_mask']
    if not log_probs.is_nested or not mask.is_nested:
        raise ValueError('HSL trainer uses the existing no-padding nested format')
    # Model token at i predicts target i+1 within the SAME action record.
    shifted = torch.roll(mask.values(), shifts=-1, dims=0).clone()
    shifted[mask.offsets()[1:] - 1] = 0
    loss = -(log_probs.values() * shifted).sum() * data['dp_size']
    # FSDP averages rank gradients; dp_size restores the global weighted sum.
    # The engine sums microbatch gradients before the single optimizer step.
    metrics = {}
    if 'sampling_draw_id' in data:
        ids = data['sampling_draw_id'].values()
        valid = (shifted > 0) & (ids >= 0)
        # Fixed 32 trajectory draws. No mixture coefficient enters reward NLL.
        nll = torch.zeros(feedback_slots, device=ids.device, dtype=torch.float32)
        tokens = torch.zeros_like(nll)
        nll.scatter_add_(0, ids[valid].long(), -log_probs.values().detach()[valid].float())
        tokens.scatter_add_(0, ids[valid].long(), torch.ones_like(ids[valid], dtype=torch.float32))
        values = torch.stack((nll, tokens)).cpu().tolist()
        metrics = {f'odm_{name}_{i}': value for name, array in zip(('nll', 'tokens'), values)
                   for i, value in enumerate(array)}
    return loss, metrics
