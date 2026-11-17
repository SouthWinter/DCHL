"""Train the valid alternative candidate once; isolate proposal construction value."""
import json
import torch.distributed as dist
from train_alfworld_batch_selection import SelectionTrainer, SelectionBuffer, parent, choose_jobs, evidence_entry


class DirectCandidateTrainer(SelectionTrainer):
    def choose_batch_update(self, package, buffer, demos, config, meta, step, root, data_a, ds_a, lengths_a):
        rank=dist.get_rank()
        payload,proposal=[None],None
        if rank==0:
            try:
                proposal=buffer.propose(package,step,int(self.config.trainer.seed))
                if proposal:
                    payload[0]=dict(rows=proposal['rows'],provenance=proposal['provenance'])
            except Exception as exc:
                payload[0]=dict(error=f'{type(exc).__name__}: {exc}')
        dist.broadcast_object_list(payload,src=0)
        item=payload[0]
        if item and 'error' in item:
            raise RuntimeError(item['error'])
        if item:
            data,ds,lengths=self.prepare_batch(item['rows'],config,meta)
        else:
            data,ds,lengths=data_a,ds_a,lengths_a
        output=self.training_client.train_batch(data=data)
        if item:
            package.update(item)
            if rank==0:
                buffer.commit_proposal(proposal)
        report=dict(selected='B' if item else 'A',reason='direct_candidate_without_update_selection',
                    updates_executed=1,updates_committed=1,reference_tokens=0)
        if rank==0:
            (root/'rounds'/f'{step:05d}'/'direct_candidate.json').write_text(json.dumps(report,indent=2)+'\n')
        return output,ds,lengths,report


if __name__=='__main__':
    parent.choose_jobs=choose_jobs
    parent.recovery_entry=evidence_entry
    online=parent.online
    online.HSLBuffer=SelectionBuffer
    online.mixed_records=parent.mixed_records
    online.ODMController=parent.BufferValueController
    online.OnlineHSLTrainer=DirectCandidateTrainer
    online.main()
