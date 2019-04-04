from sklearn.preprocessing import MinMaxScaler

from generator.generator import BrownianGenerator
from transformer.transformer import SlidingWindow, MinMaxTransform
from collections import deque
from enum import Enum
import numpy as np


class Actions(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class MarketEnv:

    def __init__(self, num_samples=1000, window=30):
        self.samples = BrownianGenerator(num_samples).generate()

        # self.samples = MinMaxTransform().transform(data)
        self.states = SlidingWindow(window).transform(self.samples)
        self.queue = deque(self.states)
        self.shares = 0
        self.cash = 100000.
        self.investment = self.cash

    def reset(self):
        self.shares = 0
        self.cash = 100000.
        self.queue = deque(self.states)
        return self.queue.popleft()

    def step(self, action, state):
        # 0 - Hold, 1 - Buy, 2-Sell
        price = state[-1]

        if len(self.queue) == 0:
            raise Exception("Expected states in queue, but none are left!")
        else:
            new_state = self.queue.popleft()
            # for a data array of prices with window length
            # Agents buy for last price known
            # But the evaluation needs to be on the next price, which shows
            # how good the decision was...
            # Remark: maybe a better solution for reward would be
            # to compute returns for all prices afterwards
            # and avg the reward?
            # Remark 2: maybe only for the window?
            # reward_price = new_state[0]
            price_window = new_state

        if action == Actions.SELL.value and self.shares > 0:
            self.cash = self.shares * price
            self.shares = 0

        if action == Actions.BUY.value and self.cash > price:
            part = int(self.cash / price)
            self.shares = self.shares + part
            self.cash = self.cash - part * price

        if action == Actions.HOLD.value:
            pass

        portfolio_value = price_window * self.shares + self.cash
        returns = (portfolio_value / self.investment) - np.ones(len(price_window))
        running_reward = 0.
        for r in returns:
            running_reward = 0.05 * r + (1 - 0.05) * running_reward

        return new_state, running_reward

    def __len__(self):
        return len(self.states)-1

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.samples)
        plt.show()
