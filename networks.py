import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiMLP(nn.Module):
    def __init__(self, state_sz, n_head):
        super().__init__()
        


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.mu_lin = nn.Linear(128, action_dim)
        self.sigma_lin = nn.Linear(128, action_dim)
        
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
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        
    def get_value(self, state):
        return self.model(state)

