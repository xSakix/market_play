from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.distributions import Categorical
from itertools import count
from torch.nn import init
from market_env.market_env import MarketEnv, Actions


# src: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
def weights_init(m):
    print('Following weights will be initialized as xavier uniform...')
    for name, param in m.named_parameters():
        if 'lstm' in name or 'gru' in name:
            print(name)
            if 'bias' in name:
                init.constant_(param, 0)
            else:
                init.xavier_uniform_(param)


class Policy(nn.Module):
    def __init__(self, window=30):
        super(Policy, self).__init__()
        self.hidden_size = window - 1
        # long term memory
        self.lstm = nn.LSTM(self.hidden_size, window - 1)
        # short term
        self.gru = nn.GRU(self.hidden_size, window - 1)

        self.dropout = nn.Dropout(0.5)
        self.affine1 = nn.Linear(2 * self.hidden_size, 512)
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

        out = self.dropout(out)

        x = F.relu(self.affine1(F.relu(out)))
        return F.softmax(self.affine2(x), dim=1)


class Agent:
    def __init__(self, gamma=0.99, load_existing=True, window=30, model_name='market_agent.pt'):
        self.PENALTY = -0.00001
        self.log_interval = 10
        if load_existing:
            self.policy = torch.load(model_name)
        else:
            self.policy = Policy(window)
            weights_init(self.policy)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4, betas=(0.8, 0.99), weight_decay=1e-3)
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = gamma
        self.window = window
        self.rewards = []
        self.training_loss = []
        self.portfolio_values = []

    def select_action(self, state, prices, evaluate=False):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if evaluate:
            self.policy.eval()

        probs = self.policy(state)
        if not evaluate:
            probs2 = probs.clone()
            can_exec = [self.env.can_execute_action(a.value, prices) for a in Actions]
            # print(can_exec)
            for i in range(len(can_exec)):
                if not can_exec[i]:
                    probs2[0][i] = 0.

            m = Categorical(probs2)
        else:
            m = Categorical(probs)

        action = m.sample()
        # print(probs2)
        # print('-' * 80)

        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        print(rewards.mean().item(), ' +/- ', rewards.std().item())
        for log_prob, R in zip(self.policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        print('LOSS:', policy_loss.item())
        self.training_loss.append(policy_loss.item())
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def train(self, episodes=10, num_samples=1000):
        print('Starting...')
        self.env = MarketEnv(num_samples, self.window)
        for i_episode in range(episodes):
            print('Starting episode {}'.format(i_episode))

            state, prices, ep_reward = self.env.reset()

            ep_reward, state, prices = self.run_episode(ep_reward, state, prices)
            self.rewards.append(ep_reward)
            portfolio_value = prices[-1] * self.env.shares + self.env.cash
            self.portfolio_values.append(portfolio_value)
            print(
                'Episode {}\tLast reward: {:.6f}\tAverage reward: {:.6f}\tShares:{}\tCash:{:.2f}\tPortfolio:{:.2f}'.format(
                    i_episode, ep_reward, np.mean(self.rewards), self.env.shares, self.env.cash, portfolio_value))
            self.finish_episode()

            torch.save(self.policy, 'market_agent.pt')

    def run_episode(self, ep_reward, state, prices):
        actions = []

        for t in range(len(self.env)):  # Don't infinite loop while learning
            action = self.select_action(state, prices)
            state, prices, reward, action = self.env.step(action, state, prices)
            actions.append(action)
            self.policy.rewards.append(reward)
            # ep_reward[ep_reward == 0.] = self.PENALTY
            if ep_reward == 0.:
                ep_reward = self.PENALTY
            ep_reward += reward

        c = Counter(actions)
        for k, v in c.items():
            print(Actions(k), ':', v)

        return ep_reward / len(self.env), state, prices

    def plot_results(self):
        import matplotlib.pyplot as plt
        print(self.rewards)
        for reward in self.rewards:
            plt.plot(reward)
        plt.title('rewards')
        plt.show()

    def save_training_loss(self):
        np.save("training_loss.npy", np.array(self.training_loss))

    def save_rewards(self):
        np.save("training_rewards.npy", np.array(self.rewards))

    def save_portfolios(self):
        np.save("portfolios.npy", np.array(self.portfolio_values))
