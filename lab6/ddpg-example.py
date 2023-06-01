'''DLP DDPG Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        ## TODO ##
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        ## TODO ##
        self.fc1 = nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], action_dim)

    def forward(self, x):
        ## TODO ##
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 因為是continuous action space且action 皆為0-1，所以最後一層不用relu # 
        x  = torch.tanh(self.fc3(x))
        return x


class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
        )

    def forward(self, x, action):
        x = self.critic_head(torch.cat([x, action], dim=1))
        return self.critic(x)


class DDPG:
    def __init__(self, args):
        # behavior network
        self._actor_net = ActorNet().to(args.device)
        self._critic_net = CriticNet().to(args.device)
        # target network
        self._target_actor_net = ActorNet().to(args.device)
        self._target_critic_net = CriticNet().to(args.device)
        # initialize target network
        '''
        将self._actor_net的权重和偏置复制到self._target_actor_net中，以实现目标网络的更新。
        这是深度强化学习中常用的技术，用于减少目标网络和行动网络之间的偏差，从而提高训练效果。
        用target net是因為它的参数不会在训练过程中更新，而是定期从行动网络中复制过来。
        这样，我们就可以使用一个稳定的目标网络来计算目标值，从而提高训练的稳定性和效果。
        '''
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net.load_state_dict(self._critic_net.state_dict())
        ## TODO ##
        # self._actor_opt = ?
        # self._critic_opt = ?
        # use adam optimizer import torch.optim as optim
        self._actor_opt = optim.Adam(self._actor_net.parameters(), lr=args.lra)
        self._critic_opt = optim.Adam(self._critic_net.parameters(), lr=args.lrc)

        # action noise
        self._action_noise = GaussianNoise(dim=2)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma

    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        ## TODO ##
        # state = torch.from_numpy(state).float().cuda() 和下面一樣意思
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        # 因為net皆要在cuda上面運算，所以state也要轉成cuda
        select_action = self._actor_net(state).cpu().data.numpy()
        if noise:
            select_action += self._action_noise.sample()

        return select_action
    
    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, [reward / 100], next_state,
                            [int(done)])

    def update(self):
        # update the behavior networks
        # 址更新裡面的optimizer參數
        self._update_behavior_network(self.gamma)
        # update the target networks
        self._update_target_network(self._target_actor_net, self._actor_net,
                                    self.tau)
        self._update_target_network(self._target_critic_net, self._critic_net,
                                    self.tau)

    def _update_behavior_network(self, gamma):
        actor_net, critic_net, target_actor_net, target_critic_net = self._actor_net, self._critic_net, self._target_actor_net, self._target_critic_net
        actor_opt, critic_opt = self._actor_opt, self._critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## update critic ##
        # critic loss
        ## TODO ##
        # q_value = ?
        # with torch.no_grad():
        #    a_next = ?
        #    q_next = ?
        #    q_target = ?
        # criterion = ?
        # critic_loss = criterion(q_value, q_target)
        q_value = critic_net(state, action)
        '''
        使用 torch.no_grad() 上下文管理器可以暫時關閉計算圖的構建，
        從而節省內存並提高計算效率。只有這個功能嗎??????
        '''
        with torch.no_grad():
            # use actor_net to get next action and input to target_critic_net
            # 可看圖
            a_next = target_actor_net(next_state)
            q_next = target_critic_net(next_state, a_next)
            q_target = reward + gamma * q_next * (1 - done)
        criterion = nn.MSELoss()
        critic_loss = criterion(q_value, q_target)
        # raise NotImplementedError

        # optimize critic
        actor_net.zero_grad()
        critic_net.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        ## update actor ##
        # actor loss
        ## TODO ##
        # action = ?
        # actor_loss = ?
        # 利用當下的state去算出action，再利用critic_net去算出q_value
        '''
        Critic網絡則評估該動作的價值，然後更新Actor網絡的參數，
        以使該動作產生更高的回報。值函數的更新是基於Q-learning，
        使用Bellman方程進行更新。
        '''
        
        action = actor_net(state)
        actor_loss = -critic_net(state, action).mean()
        # raise NotImplementedError

        # optimize actor
        actor_net.zero_grad()
        critic_net.zero_grad()
        actor_loss.backward()
        actor_opt.step()

    @staticmethod
    def _update_target_network(target_net, behavior_net, tau):
        '''update target network by _soft_ copying from behavior network'''
        '''
        软复制是一种更新目标网络的方法，它不是直接将行动网络的参数复制到目标网络中，
        而是使用一种平滑的方式将行动网络的参数与目标网络的参数混合。这种方法可以使目标网络的更新更加平滑，从而提高训练的稳定性和效果。
        '''
        for target, behavior in zip(target_net.parameters(), behavior_net.parameters()):
            ## TODO ##
            target.data.copy_(tau * behavior.data + (1 - tau) * target.data)
            # raise NotImplementedError

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                    'target_actor': self._target_actor_net.state_dict(),
                    'target_critic': self._target_critic_net.state_dict(),
                    'actor_opt': self._actor_opt.state_dict(),
                    'critic_opt': self._critic_opt.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net.load_state_dict(model['critic'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net.load_state_dict(model['target_critic'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt.load_state_dict(model['critic_opt'])


def train(args, env, agent, writer):
    print('Start Training')
    total_steps = 0
    ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition (st, at, rt, st+1) in R
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update()

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward,
                            ewma_reward))
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        ## TODO ##
        # ...
        #     if done:
        #         writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
        #         ...
        while True:
            if args.render:
                env.render() # 可視化
            
            action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition (st, at, rt, st+1) in R
            agent.append(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                rewards.append(total_reward)
                print('total_reward:',total_reward)
                break

        # raise NotImplementedError
    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='ddpg.pth')
    parser.add_argument('--logdir', default='log/ddpg')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--capacity', default=500000, type=int)
    parser.add_argument('--lra', default=1e-3, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--tau', default=.005, type=float)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLanderContinuous-v2')
    agent = DDPG(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
