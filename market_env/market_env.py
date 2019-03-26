from generator import generator
from transformer.transformer import SlidingWindow
from collections import deque
from enum import Enum


class Actions(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class MarketEnv:

    def __init__(self):
        self.samples = generator.BrownianGenerator(1000).generate()
        self.states = SlidingWindow(30).transform(self.samples)
        self.queue = deque(self.states)
        self.shares = 0
        self.cash = 100000.

    def reset(self):
        self.shares = 0
        self.cash = 100000.
        self.queue = deque(self.states)
        return self.queue.popleft()

    def step(self, action, state):
        # 0 - Hold, 1 - Buy, 2-Sell
        price = state[-1]
        last_price = state[-2]

        last_portfolio_value = last_price * self.shares + self.cash

        if action == Actions.SELL.value and self.shares > 0:
            self.cash = self.shares * price
            self.shares = 0

        if action == Actions.BUY.value and self.cash > price:
            part = int(self.cash / price)
            self.shares = self.shares + part
            self.cash = self.cash - part * price

        if action == Actions.HOLD.value:
            pass

        current_portfolio_value = price * self.shares + self.cash
        reward = (current_portfolio_value / last_portfolio_value) - 1.

        if len(self.queue) == 0:
            new_state = self.reset()
        else:
            new_state = self.queue.popleft()

        return new_state, reward
