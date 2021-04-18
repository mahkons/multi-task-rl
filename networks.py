import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadMLP(nn.Module):
    def __init__(self, in_sz, state_sz, num_head):
        super().__init__()
        self.num_head = num_head
        self.linear1 = [nn.Linear(in_sz, state_sz) for _ in range(num_head)] # TODO better implementation?
        self.linear2 = [nn.Linear(state_sz, state_sz) for _ in range(num_head)] # TODO better implementation?

        for i, l in enumerate(self.linear1):
            self.add_module("lin1_{}".format(i), l)

        for i, l in enumerate(self.linear2):
            self.add_module("lin2_{}".format(i), l)

    # input (batch, *)
    def forward(self, x):
        xs = [self.linear2[i](F.elu(self.linear1[i](x))) for i in range(self.num_head)]
        return torch.transpose(torch.stack(xs), 0, 1) # output (batch, heads, *)
        

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            MultiHeadMLP(256, 64, 5),
            nn.Flatten(start_dim=1),
            nn.ELU(),
        )
        self.mu_lin = nn.Linear(64*5, action_dim)
        self.sigma_lin = nn.Linear(64*5, action_dim)
        
    def compute_proba(self, state, action):
        x = self.model(state)
        mu, sigma = self.mu_lin(x), self.sigma_lin(x)
        sigma = torch.exp(sigma)
        distr = torch.distributions.Normal(mu, sigma)
        return distr.log_prob(action).sum(axis=1), distr
        
    def act(self, state):
        x = self.model(state)
        mu, sigma = self.mu_lin(x), self.sigma_lin(x)
        sigma = torch.exp(sigma)
        distr = torch.distributions.Normal(mu, sigma)
        pure_action = distr.sample()
        action = torch.tanh(pure_action)
        return action, pure_action, distr.log_prob(pure_action).sum(axis=1)
        
        
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            MultiHeadMLP(256, 64, 5),
            nn.ELU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 5, 1),
        )
        
    def get_value(self, state):
        return self.model(state)

