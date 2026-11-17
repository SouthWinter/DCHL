"""Bounded shared-memory weight transfer from FSDP2 to a resident vLLM worker.

All training ranks iterate the existing FSDP parameter generator. Rank zero
transfers one CPU bucket through /dev/shm; no model checkpoint is written to disk.
The training loop must finish all rollouts before calling sync, and start the
next collection only after commit returns the requested policy version.
"""
from pathlib import Path
import tempfile
import time
import json
import urllib.request

import torch


class HSLWeightWorker:
    """vLLM worker extension, compatible with the installed 0.8.5 V1 API."""
    def hsl_begin(self, version):
        self._hsl_pending = version
        self._hsl_seen = set()
        return {'pending_version': version}

    def hsl_load_bucket(self, path, version):
        if getattr(self, '_hsl_pending', None) != version:
            raise ValueError('weight bucket does not match pending policy version')
        weights = torch.load(path, map_location='cpu', weights_only=True)
        if self._hsl_seen.intersection(weights):
            raise ValueError('duplicate source tensor in weight transfer')
        # Qwen's native loader handles q/k/v and gate/up fused parameter mapping.
        self.model_runner.model.load_weights(weights.items())
        self._hsl_seen.update(weights)
        return {'version': version, 'source_tensors_loaded': len(weights)}

    def hsl_commit(self, version, tensor_count):
        if getattr(self, '_hsl_pending', None) != version or len(self._hsl_seen) != tensor_count:
            raise ValueError('incomplete weight transfer')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._hsl_version = version
        self._hsl_pending = None
        return {'policy_version': version, 'source_tensor_count': tensor_count}

    def hsl_policy_version(self):
        return {'policy_version': getattr(self, '_hsl_version', None),
                'pending_version': getattr(self, '_hsl_pending', None)}

    def hsl_probe_snapshot(self):
        """Opt-in value check; normal training never allocates this snapshot."""
        self._hsl_probe_reference = {
            name: value.detach().cpu().clone()
            for name, value in self.model_runner.model.named_parameters()}
        return {'native_tensors': len(self._hsl_probe_reference),
                'reference_bytes': sum(v.numel()*v.element_size()
                                       for v in self._hsl_probe_reference.values())}

    def hsl_probe_compare(self):
        reference = self._hsl_probe_reference
        records = []
        for name, value in self.model_runner.model.named_parameters():
            old = reference.pop(name)
            current = value.detach().cpu()
            equal = torch.equal(current, old)
            records.append(dict(name=name, shape=list(current.shape), dtype=str(current.dtype),
                equal=equal, max_abs_error=0.0 if equal else
                    float((current.float()-old.float()).abs().max())))
        if reference:
            raise ValueError('Native parameter names changed during synchronization')
        del self._hsl_probe_reference
        return dict(status='MATCH' if all(r['equal'] for r in records) else 'DIFFERENCE_FOUND',
                    records=records, tensors=len(records))


def request(endpoint, method, payload):
    request = urllib.request.Request(endpoint.rstrip('/') + '/hsl/' + method,
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def sync_engine_weights(engine, endpoint, version, bucket_megabytes=256):
    import torch.distributed as dist
    rank = dist.get_rank()
    params, peft_config = engine.get_per_tensor_param()
    if peft_config is not None:
        raise ValueError('this shared HSL comparison uses full-parameter models')
    started = time.monotonic()
    if rank == 0:
        request(endpoint, 'begin', {'version': version})
        scratch = tempfile.TemporaryDirectory(prefix='alfworld-hsl-weights.', dir='/dev/shm')
    bucket, size, count, transferred = {}, 0, 0, 0

    def flush():
        nonlocal bucket, size, transferred
        if rank == 0 and bucket:
            path = Path(scratch.name) / 'bucket.pt'
            torch.save(bucket, path)
            request(endpoint, 'load_bucket', {'path': str(path), 'version': version})
            path.unlink()
            transferred += size
            bucket, size = {}, 0

    try:
        for name, tensor in params:
            # Generator invokes FSDP collectives on every rank in identical order.
            count += 1
            if rank == 0:
                tensor = tensor.detach().to(device='cpu', dtype=torch.bfloat16)
                tensor_bytes = tensor.numel() * tensor.element_size()
                if size and size + tensor_bytes > bucket_megabytes * 1024**2:
                    flush()
                bucket[name] = tensor
                size += tensor_bytes
        flush()
        result = None
        if rank == 0:
            result = request(endpoint, 'commit', {'version': version, 'tensor_count': count})
            if result['policy_version'] != version:
                raise ValueError('service did not acknowledge requested policy version')
            result.update(weight_bytes=transferred, sync_seconds=time.monotonic() - started)
        dist.barrier()
        return result
    finally:
        if rank == 0:
            scratch.cleanup()
