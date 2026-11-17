"""Beta-actor PPO top-k selection for online, verified goal candidates."""
import base64
import io
import json
import math
import random

import torch
from torch import nn
from torch.distributions import Beta

import alfworld_batch_selection as selection
from alfworld_stable_evidence import source_key


class MLP(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output)

    def forward(self, x):
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class Policy:
    def __init__(self, seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = MLP(2)
            self.critic = MLP(1)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4, eps=1e-5, weight_decay=5e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4, eps=1e-5, weight_decay=5e-4)
        self.records = []
        self.updates = 0
        self.seed = seed

    def act(self, states, step):
        x = torch.tensor(states, dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + step * 8191)
            with torch.no_grad():
                parameters = torch.nn.functional.softplus(self.actor(x)) + 1.
                dist = Beta(parameters[:, 0], parameters[:, 1])
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = self.critic(x).squeeze(-1)
        return x, actions, log_probs, values

    def add(self, states, actions, log_probs, values, rewards):
        self.records.append(tuple(t.detach().cpu() for t in
                            (states, actions, log_probs, values, rewards)))

    def update(self, step=None, total_steps=None):
        if not self.records:
            return None
        if step is not None and total_steps is not None:
            factor = 1 - .99*(step-1)/max(1,total_steps-1)
            for optimizer in (self.actor_optim,self.critic_optim):
                for group in optimizer.param_groups:
                    group['lr'] = 3e-4*factor
        x, actions, old_logp, old_values = (torch.cat([r[i] for r in self.records])
                                             for i in range(4))
        next_value = torch.zeros_like(self.records[-1][3])
        gae = torch.zeros_like(next_value)
        advantages_by_round = []
        for _, _, _, values, rewards in reversed(self.records):
            delta = rewards + .99*next_value-values
            gae = delta + .99*.95*gae
            advantages_by_round.insert(0,gae)
            next_value = values
        advantages = torch.cat(advantages_by_round)
        returns = advantages + old_values
        advantages = advantages/(advantages.std(unbiased=False)+1e-8)
        for _ in range(4):
            parameters = torch.nn.functional.softplus(self.actor(x))+1.
            dist = Beta(parameters[:, 0], parameters[:, 1])
            ratio = (dist.log_prob(actions)-old_logp).exp()
            actor_loss = -torch.minimum(ratio*advantages,
                ratio.clamp(.8,1.2)*advantages).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            critic_loss = torch.nn.functional.mse_loss(self.critic(x).squeeze(-1), returns)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
        self.records.clear()
        self.updates += 1
        return dict(actor_loss=float(actor_loss), critic_loss=float(critic_loss),
                    transitions=len(actions), ppo_updates=self.updates)

    def serialize(self):
        stream = io.BytesIO()
        torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
            actor_optim=self.actor_optim.state_dict(), critic_optim=self.critic_optim.state_dict(),
            records=self.records, updates=self.updates, seed=self.seed), stream)
        return base64.b64encode(stream.getvalue()).decode('ascii')

    @classmethod
    def deserialize(cls, value):
        state = torch.load(io.BytesIO(base64.b64decode(value)), map_location='cpu', weights_only=False)
        policy = cls(state['seed'])
        policy.actor.load_state_dict(state['actor'])
        policy.critic.load_state_dict(state['critic'])
        policy.actor_optim.load_state_dict(state['actor_optim'])
        policy.critic_optim.load_state_dict(state['critic_optim'])
        policy.records = state['records']
        policy.updates = state['updates']
        return policy


class DataAgentBuffer(selection.SelectionBuffer):
    """Online FIFO identity, policy state and a 50%-ratio candidate tranche."""
    feedback_metric_key = 'data_agent_selection'
    vocab_size = 128256

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = None
        self.scores = {}
        self.last_data_agent = {}

    def candidates(self, step, seed):
        entries = list(self.demonstrations)
        if not entries:
            return []
        rng = random.Random(seed+step*7919+101)
        weights = [max(.001,e['goal']['hsl_weight']) for e in entries]
        # Two candidates per available supervision slot; duplicate draws still
        # receive independent policy actions, as in minibatch sample selection.
        return rng.choices(entries, weights=weights, k=64)

    @staticmethod
    def identity(entry):
        return json.dumps([source_key(entry), entry['goal']['instruction'],
            entry['goal']['prefix_length'],entry['goal'].get('feedback_steps')],
            separators=(',',':'),ensure_ascii=False)

    def scoring_rows(self, entries):
        rows = []
        for index, entry in enumerate(entries):
            goal = entry['goal']
            source = selection.training_records(entry['rollout'], goal, .3, 32)
            feedback = set(goal['feedback_steps'])
            selected = [r for r in source if r['action_supervised'] and r['step'] in feedback]
            if not selected:
                raise ValueError('Verified candidate lacks feedback action')
            row = selected[-1]
            row['data_agent_candidate_idx'] = index
            row['sampling_draw_id'] = -1
            rows.append(row)
        return rows

    @classmethod
    def state_from_scores(cls, nll, entropy, nll_std, entropy_std, count):
        return [min(10., math.log1p(max(0.,nll))),
                min(1., max(0.,entropy/math.log(cls.vocab_size))),
                min(10., math.log1p(max(0.,nll_std))),
                min(1., max(0.,entropy_std/math.log(cls.vocab_size))),
                min(10., math.log1p(max(0.,count))),
                min(10., math.log1p(max(0.,nll*count)))]

    def select(self, entries, states, metrics, package, step, seed, total_steps):
        if self.policy is None:
            self.policy = Policy(seed)
        x, actions, log_probs, values = self.policy.act(states, step)
        scores = actions.tolist()
        for entry, score in zip(entries,scores):
            self.scores[self.identity(entry)] = score
        # Official schedule: initial warmup and final 12.5% without pruning.
        if step == 1 or step > math.floor(.875*total_steps):
            chosen = random.Random(seed+step*15401).sample(range(len(entries)),32)
            mode = 'uniform_schedule'
        else:
            chosen = sorted(range(len(entries)),key=lambda i:(scores[i],-i),reverse=True)[:32]
            mode = 'top_half_beta_score'
        nll = torch.tensor([m['nll'] for m in metrics],dtype=torch.float32)
        entropy = torch.tensor([m['entropy'] for m in metrics],dtype=torch.float32)
        nll = nll.clamp(max=float(torch.quantile(nll,.95)))
        nll = (nll-nll.min())/(nll.max()-nll.min()+1e-8)
        entropy = entropy/math.log(self.vocab_size)
        weight = nll.var(unbiased=False)/(nll.var(unbiased=False)+entropy.var(unbiased=False)+1e-8)
        rewards = (weight*nll+(1-weight)*entropy)*actions.detach()
        rewards = (rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-8)
        self.policy.add(x,actions,log_probs,values,rewards)
        ppo = self.policy.update(step,total_steps) if step%10==0 else None
        aux_count = sum(p['actions'] for p in package['provenance']
                        if p['source']=='verified_recovery')
        cut=len(package['rows'])-aux_count
        rows=[r for r in package['rows'][:cut] if 'policy_version' not in r]
        auxiliary=package['rows'][cut:]
        assert rows and all('policy_version' in r for r in auxiliary)
        provenance=[p for p in package['provenance'] if p['source']=='original']
        for draw,index in enumerate(chosen):
            entry=entries[index]
            goal,rollout=entry['goal'],entry['rollout']
            records=[r for r in selection.training_records(rollout,goal,.3,32) if r['action_supervised']]
            feedback=set(goal['feedback_steps'])
            for row in records:
                row['sampling_draw_id']=draw if row['step'] in feedback else -1
            rows.extend(records)
            provenance.append(dict(source='hindsight',task=rollout['gamefile'],
                sampling_draw_id=draw,goal=goal['instruction'],prefix_length=goal['prefix_length'],
                supervised_actions=goal['relevant_actions'],feedback_steps=goal['feedback_steps'],
                feedback_actions=sum(r['sampling_draw_id']==draw for r in records),
                policy_version=rollout['policy_version'],candidate_id=self.identity(entry),
                allocation='Data Agent Beta actor score, top-half candidate selection'))
        random.Random(seed+step*20311).shuffle(rows)
        rows.extend(auxiliary)
        provenance.extend(p for p in package['provenance'] if p['source']=='verified_recovery')
        self.last_data_agent=dict(mode=mode, candidate_count=len(entries),draws=len(chosen),
            candidate_ids=[self.identity(e) for e in entries],selected_indices=chosen,
            scores=scores, nll=[m['nll'] for m in metrics],
            entropy=[m['entropy'] for m in metrics],reward_weight=float(weight),ppo=ppo)
        return dict(rows=rows,provenance=provenance,selected=[entries[i] for i in chosen])

    def commit_proposal(self, proposal):
        self.last_selected=proposal['selected']
        self.last_sample_distribution={}

    def observe_training(self, metrics):
        # Candidate NLL and entropy are measured before the update and are
        # already consumed by the Beta/PPO policy.  The inherited rehearsal
        # callback expects per-draw ODM metrics that this selector does not
        # request from the trainer.
        return dict(mode=self.last_data_agent.get('mode'),
                    candidates=self.last_data_agent.get('candidate_count', 0),
                    selected=self.last_data_agent.get('draws', 0),
                    policy_updates=self.policy.updates if self.policy else 0)

    def state_dict(self):
        return dict(super().state_dict(),data_agent_policy=self.policy.serialize() if self.policy else None,
                    data_agent_scores=self.scores,data_agent_last=self.last_data_agent)

    @classmethod
    def from_state_dict(cls,state):
        buffer=super().from_state_dict(state)
        buffer.policy=Policy.deserialize(state['data_agent_policy']) if state.get('data_agent_policy') else None
        buffer.scores=state.get('data_agent_scores',{})
        buffer.last_data_agent=state.get('data_agent_last',{})
        return buffer
