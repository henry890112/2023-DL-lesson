'''DLP DQN Lab'''
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
import torch.nn.functional as F
import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        # deque為雙向佇列，可從頭尾兩端append和pop
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        '''
        transition通常是一個包含當前狀態、動作、獎勵、下一個狀態等信息的元組。在這個程式碼中，
        transition被轉換為一個tuple，其中每個元素也是一個元組。這是因為在Python中，元組是不可變的數據結構，
        可以更好地保證緩衝區中的數據不會被修改。最後，這個轉換被添加到緩衝區的末尾，使用了deque的append方法。
        '''
        # 將裡面的每個元素也都轉換成tuple
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=32):
        super().__init__()
        ## TODO ##
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim* 2)
        self.fc3 = nn.Linear(hidden_dim* 2, action_dim)
        
    def forward(self, x):
        ## TODO ##
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        ## TODO ##
        # use adam optimizer import torch.optim as optim
        self._optimizer = optim.Adam(self._behavior_net.parameters(), lr=args.lr)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        '''
        用此可以得到action_space的action
        NUM_ACTIONS = env.action_space.n # (left, right)
        '''
         ## TODO ##
         # Select a random action
         ##### why it cannot work!!!!!!!!!!!!!!!!!!!
        # if random.random() < epsilon:
        #     action = np.random.randint(action_space.n)
        #     # action = action_space.sample()
        # # Select the action with the highest q
        # else:
        #     state = torch.from_numpy(state).float().unsqueeze(0).cuda()  # 增加一個維度
     
        rnd = random.random()
        if rnd < epsilon:
            return np.random.randint(action_space.n)
        else:           
            state = torch.from_numpy(state).float().unsqueeze(0).cuda()
            with torch.no_grad():
                actions_value = self._behavior_net.forward(state)#state as input out put is action
            action = np.argmax(actions_value.cpu().data.numpy()) #take max q action as action
        return action

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state,
                            [int(done)])

    def update(self, total_steps):
        # freq = 4 
        # target_freq = 100
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## TODO ##
        # q_value = ?
        # with torch.no_grad():
        #    q_next = ?
        #    q_target = ?
        # criterion = ?
        # loss = criterion(q_value, q_target)

        # get q value of action
        '''
        .gather(dim, index)
        是PyTorch中的一个张量方法，用于从指定维度上选择指定索引的元素
        '''
        # use .gather(dim, index) to get q value of action, 因為每列取一個值, 所以shape(64,1)
        q_value = self._behavior_net(state).gather(1, action.long())  
        with torch.no_grad():
            # use target net to get next q value, more stable
            # get next q value and index, and use [0] to get value
            q_next = self._target_net(next_state).max(1)[0]
            # 用view轉換維度
            q_target = reward +self.gamma * q_next.view(self.batch_size, 1) #shape(64,1)            print('reward:',reward,'q_value:',q_value,'q_target:',q_target)
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target) 

        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent, writer):
    print('Start Training')
    '''
    ## Check dimension of spaces ##
    print(env.action_space)
    #> Discrete(2)
    print(env.observation_space)
    #> Box(4,)
    ## Check range of spaces ##
    '''
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()  # reset() returns initial observation
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                # 隨機選擇一個action去執行，warmup期間不更新網路
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # execute action
            '''
            由 step function 提供環境狀態。step 會回傳 4 個變數：
                observation (環境狀態)
                reward (上一次 action 獲得的 reward )
                done (判斷是否達到終止條件的變數)
                info ( debug 用的資訊)
            '''
            # print(env.step(action))
            # print(env.step(action)[4])
            next_state, reward, done, _= env.step(action)  
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

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
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
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
            
            action = agent.select_action(state, epsilon, action_space)
            epsilon = max(epsilon * args.eps_decay, args.eps_min)# selection
            next_state, reward, done, _ = env.step(action)# execute
            agent.append(state, action, reward, next_state, done) # transition

            state = next_state
            total_reward += reward

            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                rewards.append(total_reward)
                print('total_reward:',total_reward)
                break

    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='dqn.pth')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2')
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
