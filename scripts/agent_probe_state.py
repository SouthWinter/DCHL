"""Reversible local-shard snapshots for AdamW supervision lookahead.

Every training rank must capture/restore at the same resharded update boundary.
Tensor identity and DTensor placement stay in the engine; only local values are
copied to CPU. This also restores the lazy-empty optimizer at the first update.
"""
import copy
import random
import numpy as np
import torch


def local(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


class TensorValue:
    def __init__(self, value):
        self.target = value
        self.saved = local(value).detach().cpu().clone()

    def restore(self):
        with torch.no_grad():
            local(self.target).copy_(self.saved)
        return self.target


def capture_tree(value):
    if isinstance(value, torch.Tensor):
        return TensorValue(value)
    if isinstance(value, dict):
        return {k:capture_tree(v) for k,v in value.items()}
    if isinstance(value, list):
        return [capture_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(capture_tree(v) for v in value)
    return copy.deepcopy(value)


def restore_tree(value):
    if isinstance(value, TensorValue):
        return value.restore()
    if isinstance(value, dict):
        return {k:restore_tree(v) for k,v in value.items()}
    if isinstance(value, list):
        return [restore_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(restore_tree(v) for v in value)
    return copy.deepcopy(value)


class AdamProbeState:
    def __init__(self, module, optimizer, scheduler=None):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError('Validated only for torch AdamW')
        self.module, self.optimizer, self.scheduler = module, optimizer, scheduler
        self.parameters = {name:TensorValue(p) for name,p in module.named_parameters()}
        self.buffers = {name:TensorValue(b) for name,b in module.named_buffers()}
        self.optimizer_state = {p:capture_tree(s) for p,s in optimizer.state.items()}
        self.groups = [{k:copy.deepcopy(v) for k,v in g.items() if k!='params'} for g in optimizer.param_groups]
        self.scheduler_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
        self.cpu_rng = torch.get_rng_state().clone()
        self.gpu_rng = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
        self.python_rng, self.numpy_rng = random.getstate(), np.random.get_state()
        self.training = {name:m.training for name,m in module.named_modules()}

    def restore(self):
        current = dict(self.module.named_parameters())
        assert current.keys() == self.parameters.keys()
        for name, snapshot in self.parameters.items():
            assert current[name] is snapshot.target, f'Parameter identity changed at {name}; snapshot boundary must be resharded'
            snapshot.restore()
        for snapshot in self.buffers.values():
            snapshot.restore()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.state.clear()
        self.optimizer.state.update({p:restore_tree(s) for p,s in self.optimizer_state.items()})
        assert len(self.groups) == len(self.optimizer.param_groups)
        for group, saved in zip(self.optimizer.param_groups, self.groups):
            params = group['params']
            group.clear()
            group.update(copy.deepcopy(saved), params=params)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(copy.deepcopy(self.scheduler_state))
        for name,m in self.module.named_modules():
            m.training = self.training[name]
        torch.set_rng_state(self.cpu_rng)
        if self.gpu_rng is not None:
            torch.cuda.set_rng_state(self.gpu_rng)
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)

    def assert_restored(self):
        for name, snapshot in self.parameters.items():
            assert torch.equal(local(snapshot.target).detach().cpu(), snapshot.saved), name
        assert set(self.optimizer.state) == set(self.optimizer_state)
        def check(now, saved):
            if isinstance(saved, TensorValue):
                assert torch.equal(local(now).detach().cpu(), saved.saved)
            elif isinstance(saved, dict):
                assert now.keys() == saved.keys()
                for k in saved:
                    check(now[k], saved[k])
            elif isinstance(saved, (list,tuple)):
                assert len(now) == len(saved)
                for a,b in zip(now,saved):
                    check(a,b)
            else:
                assert now == saved
        for p,saved in self.optimizer_state.items():
            check(self.optimizer.state[p], saved)
        for name, snapshot in self.buffers.items():
            assert torch.equal(local(snapshot.target).detach().cpu(), snapshot.saved), name
        assert [{k:v for k,v in g.items() if k!='params'} for g in self.optimizer.param_groups] == self.groups
        if self.scheduler is not None:
            assert self.scheduler.state_dict() == self.scheduler_state
        assert {name:m.training for name,m in self.module.named_modules()} == self.training
        assert torch.equal(torch.get_rng_state(), self.cpu_rng)
        if self.gpu_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(), self.gpu_rng)
        assert random.getstate() == self.python_rng
        current = np.random.get_state()
        assert current[0] == self.numpy_rng[0] and np.array_equal(current[1], self.numpy_rng[1])
        assert current[2:] == self.numpy_rng[2:]
