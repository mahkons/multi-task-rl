import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class MultiHeadMLP(nn.Module):
    def __init__(self, in_sz, hidden_sz, state_sz, num_head):
        super().__init__()
        self.num_head = num_head
        self.linear1 = [nn.Linear(in_sz, hidden_sz) for _ in range(num_head)] # TODO conv1D implementation?
        self.linear2 = [nn.Linear(hidden_sz, state_sz) for _ in range(num_head)]

        for i, l in enumerate(self.linear1):
            self.add_module("lin1_{}".format(i), l)

        for i, l in enumerate(self.linear2):
            self.add_module("lin2_{}".format(i), l)

    # input (batch, *)
    def forward(self, x):
        xs = [self.linear2[i](F.elu(self.linear1[i](x))) for i in range(self.num_head)]
        return torch.transpose(torch.stack(xs), 0, 1) # output (batch, heads, *)
        

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, num_shared):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.mlp = MultiHeadMLP(256, 64, 32, 5)
        self.fc2 = nn.Linear(32*5*(num_shared + 1), 128)
        self.mu_lin = nn.Linear(128, action_dim)
        self.sigma_lin = nn.Linear(128, action_dim)

        self.shared = list()

    def add_shared(self, module):
        self.add_module("shared_{}".format(len(self.shared)), module)
        self.shared.append(module)

    def model(self, state):
        x = F.elu(self.fc1(state))
        y = F.elu(torch.stack([sh_m(x) for sh_m in self.shared]))
        y = torch.transpose(y, 0, 1)
        x = F.elu(self.mlp(x))
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        return F.elu(self.fc2(torch.cat([x, y], dim=1)))
        
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

    def first_parameters(self):
        return self.fc1.parameters()

    def shared_parameters(self):
        return itertools.chain(*map(lambda x: x.parameters(), self.shared))

    def rest_parameters(self):
        return itertools.chain(self.mlp.parameters(), self.fc2.parameters(), self.mu_lin.parameters(), self.sigma_lin.parameters())
        
        
class Critic(nn.Module):
    def __init__(self, state_dim, num_shared):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.mlp = MultiHeadMLP(256, 64, 32, 5)
        self.fc2 = nn.Linear(32*5*(num_shared + 1), 128)
        self.fc3 = nn.Linear(128, 1)

        self.shared = list()

    def add_shared(self, module):
        self.add_module("shared_{}".format(len(self.shared)), module)
        self.shared.append(module)

    def model(self, state):
        x = F.elu(self.fc1(state))
        y = F.elu(torch.stack([sh_m(x) for sh_m in self.shared]))
        y = torch.transpose(y, 0, 1)
        x = F.elu(self.mlp(x))
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        return self.fc3(F.elu(self.fc2(torch.cat([x, y], dim=1))))
        
    def get_value(self, state):
        return self.model(state)

    def first_parameters(self):
        return self.fc1.parameters()

    def shared_parameters(self):
        return itertools.chain(*map(lambda x: x.parameters(), self.shared))

    def rest_parameters(self):
        return itertools.chain(self.mlp.parameters(), self.fc2.parameters(), self.fc3.parameters())

