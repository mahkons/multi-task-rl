import pybullet_envs
# Don't forget to install PyBullet!

import gym
import numpy as np
import torch
from torch import nn
import random
import itertools

from PPO import PPO
from logger import init_logger, log
from params import ENV_NAMES, ENV_NAMES_SHORT, MIN_EPISODES_PER_UPDATE, MIN_TRANSITIONS_PER_UPDATE, ITERATIONS, LR, FIRST_LR, SHARED_LR
from networks import MultiHeadMLP

device = torch.device("cuda")


def evaluate_policy(env, agent, episodes):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns
   

def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, logprob = agent.act(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, logprob))
        s = ns
    return trajectory


def train(env_names):
    for env_name in env_names:
        log().add_plot(env_name + "_reward", ("episode", "step", "reward"))
    envs = [gym.make(name) for name in env_names]
    agents = [PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], num_shared=len(envs) - 1, device=device) for env in envs]

    shared_modules = list()
    for i, j in itertools.combinations(range(len(agents)), 2):
        actor_shared = MultiHeadMLP(256, 64, 32, 5).to(device)
        critic_shared = MultiHeadMLP(256, 64, 32, 5).to(device)
        shared_modules.append(actor_shared)
        shared_modules.append(critic_shared)

        agents[i].actor.add_shared(actor_shared)
        agents[i].critic.add_shared(critic_shared)
        agents[j].actor.add_shared(actor_shared)
        agents[j].critic.add_shared(critic_shared)

    episodes_sampled = [0 for _ in range(len(envs))]
    steps_sampled = [0 for _ in range(len(envs))]

    # TODO parameter groups
    first_optimizer = torch.optim.Adam(itertools.chain(*[agent.first_parameters() for agent in agents]), lr=FIRST_LR)
    shared_optimizer = torch.optim.Adam(itertools.chain(*[sh_m.parameters() for sh_m in shared_modules]), lr=SHARED_LR)
    rest_optimizer = torch.optim.Adam(itertools.chain(*[agent.rest_parameters() for agent in agents]), lr=LR)
    for i in range(ITERATIONS):
        all_trajectories = [[] for _ in range(len(envs))]
        
        for j, (env, agent) in enumerate(zip(envs, agents)):
            steps_cnt = 0
            while len(all_trajectories[j]) < MIN_EPISODES_PER_UPDATE or steps_cnt < MIN_TRANSITIONS_PER_UPDATE:
                traj = sample_episode(env, agent)
                steps_cnt += len(traj)
                all_trajectories[j].append(traj)
                reward = sum([r for _, _, r, _ in traj])
                log().add_plot_point(env_names[j] + "_reward", (episodes_sampled[j] + len(all_trajectories[j]), steps_sampled[j] + steps_cnt, reward))
            episodes_sampled[j] += len(all_trajectories[j])
            steps_sampled[j] += steps_cnt

        loss_generators = [agent.update(trajectories) for agent, trajectories in zip(agents, all_trajectories)]
        for losses in zip(*loss_generators):
            first_optimizer.zero_grad()
            shared_optimizer.zero_grad()
            rest_optimizer.zero_grad()
            sum(losses).backward()
            first_optimizer.step()
            shared_optimizer.step()
            rest_optimizer.step()
        
        log().save_logs()
    log().save_logs()



def init_random_seeds(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


if __name__ == "__main__":
    init_random_seeds(23)
    init_logger("logdir", "MultiTask")
    train(ENV_NAMES_SHORT)
