import torch
import numpy as np
import itertools

from networks import Actor, Critic
from params import LR, GAMMA, LAMBDA, CLIP, ENTROPY_COEF, VALUE_COEFF, BATCHES_PER_UPDATE, BATCH_SIZE


class PPO():
    def __init__(self, state_dim, action_dim, num_shared, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.actor = Actor(state_dim, action_dim, num_shared).to(device)
        self.critic = Critic(state_dim, num_shared).to(device)

    def parameters(self):
        return itertools.chain(self.actor.parameters(), self.critic.parameters())

    def first_parameters(self):
        return itertools.chain(self.actor.first_parameters(), self.critic.first_parameters())

    def shared_parameters(self):
        return itertools.chain(self.actor.shared_parameters(), self.critic.shared_parameters())

    def rest_parameters(self):
        return itertools.chain(self.actor.rest_parameters(), self.critic.rest_parameters())

    def _calc_loss(self, state, action, old_log_prob, expected_values, gae):
        new_log_prob, action_distr = self.actor.compute_proba(state, action)
        state_values = self.critic.get_value(state).squeeze(1)

        critic_loss = ((expected_values - state_values) ** 2).mean()

        unclipped_ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(unclipped_ratio, 1 - CLIP, 1 + CLIP)
        actor_loss = -torch.min(clipped_ratio * gae, unclipped_ratio * gae).mean()

        entropy_loss = -action_distr.entropy().mean()

        return critic_loss * VALUE_COEFF + actor_loss + entropy_loss * ENTROPY_COEF


    def update(self, trajectories):
        trajectories = map(self._compute_lambda_returns_and_gae, trajectories)
        transitions = sum(trajectories, []) # Turn a list of trajectories into list of transitions

        state, action, old_log_prob, target_value, advantage = zip(*transitions)
        state = torch.from_numpy(np.array(state)).float().to(self.device)
        action = torch.from_numpy(np.array(action)).float().to(self.device)
        old_log_prob = torch.from_numpy(np.array(old_log_prob)).float().to(self.device)
        target_value = torch.from_numpy(np.array(target_value)).float().to(self.device)
        advantage = torch.from_numpy(np.array(advantage)).float().to(self.device)

        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE)
            loss = self._calc_loss(state[idx], action[idx], old_log_prob[idx], target_value[idx], advantage[idx])

            # ugly code yeah =)
            # optimization outside
            yield loss


    def _compute_lambda_returns_and_gae(self, trajectory):
        lambda_returns = []
        gae = []
        last_lr = 0.
        last_v = 0.
        for s, _, r, _ in reversed(trajectory):
            ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
            last_lr = ret
            last_v = self.get_value(s)
            lambda_returns.append(last_lr)
            gae.append(last_lr - last_v)
        
        # Each transition contains state, action, old action probability, value estimation and advantage estimation
        return [(s, a, p, v, adv) for (s, a, _, p), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]
            
            
    def get_value(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action, pure_action, log_prob = self.actor.act(state)
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], log_prob.cpu().item()

    def save(self):
        torch.save(self.actor, "agent.pkl")


