import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.distributions import Categorical
from itertools import count
from market_env.market_env import MarketEnv


class Policy(nn.Module):
    def __init__(self,window=30):
        super(Policy, self).__init__()
        self.hidden_size = window
        # long term memory
        self.lstm = nn.LSTM(self.hidden_size, window)
        # short term
        self.gru = nn.GRU(self.hidden_size, window)

        self.affine1 = nn.Linear(2*self.hidden_size, 512)
        self.affine2 = nn.Linear(512, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        h_lstm = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
        h_gru = torch.zeros(1, 1, self.hidden_size)

        x = F.normalize(x)

        out_lstm, h_lstm = self.lstm(x.view(1, 1, -1), h_lstm)
        out_gru, h_gru = self.gru(x.view(1, 1, -1), h_gru)

        out = torch.cat((out_lstm, out_gru), dim=-1).squeeze(1)

        x = F.relu(self.affine1(F.relu(out)))
        return F.softmax(self.affine2(x), dim=1)


class Agent:
    def __init__(self, gamma=0.99, load_existing=True, window=30):
        self.log_interval = 10
        if load_existing:
            self._load_existing()
        else:
            self.policy = Policy(window)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = gamma
        self.window = window

    def _load_existing(self):
        try:
            self.policy = torch.load('market_agent.pt')
        except:
            self.policy = Policy()

    def select_action(self, state, evaluate=False):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if evaluate:
            self.policy.eval()
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def train(self, episodes=10, epochs=15,num_samples=1000):
        print('Starting...')
        running_reward = 10
        self.env = MarketEnv(num_samples,self.window)
        # for i_episode in count(1):
        for i_episode in range(1, episodes + 1):

            print('Starting episode {}'.format(i_episode))

            for epoch in range(epochs):
                state, ep_reward = self.env.reset(), 0

                ep_reward = self.run_episode(ep_reward, state)

                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                print('Episode {}:{}/{}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, epoch, epochs, ep_reward, running_reward))
                self.finish_episode()

            print('Switching environment...')
            self.env = MarketEnv(num_samples,self.window)
            running_reward = 10
            torch.save(self.policy, 'market_agent.pt')

    def run_episode(self, ep_reward, state):
        for t in range(1, len(self.env)):  # Don't infinite loop while learning
            action = self.select_action(state)
            state, reward = self.env.step(action, state)
            self.policy.rewards.append(reward)
            ep_reward += reward
        return ep_reward
