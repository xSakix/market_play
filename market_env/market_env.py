from sklearn.preprocessing import MinMaxScaler

from generator.generator import BrownianGenerator
from transformer.transformer import SlidingWindow, MinMaxTransform
from collections import deque
from enum import Enum


class Actions(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class MarketEnv:

    def __init__(self):
        self.samples = BrownianGenerator(1000000).generate()

        # self.samples = MinMaxTransform().transform(data)
        self.states = SlidingWindow(30).transform(self.samples)
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
            reward_price = price
            new_state = self.reset()
        else:
            new_state = self.queue.popleft()
            reward_price = state[0]

        if action == Actions.SELL.value and self.shares > 0:
            self.cash = self.shares * price
            self.shares = 0

        if action == Actions.BUY.value and self.cash > price:
            part = int(self.cash / price)
            self.shares = self.shares + part
            self.cash = self.cash - part * price

        if action == Actions.HOLD.value:
            pass

        portfolio_value = reward_price * self.shares + self.cash
        reward = (portfolio_value / self.investment) - 1.

        return new_state, reward

    def __len__(self):
        return len(self.states)
