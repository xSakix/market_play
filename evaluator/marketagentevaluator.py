import numpy as np
from market_env.market_env import Actions
from transformer.transformer import SlidingWindow
import matplotlib.pyplot as plt


class MarketAgentEvaluator:

    def __init__(self, agent):
        self.agent = agent
        self.shares = 0
        self.cash = 100000.
        self.portfolio = []

    def evaluate(self, data, window=30):
        prices = SlidingWindow(window).transform(data)
        states = np.log(prices[:, 1::]) - np.log(prices[:, :-(window - 1)])
        for price_window, state in zip(prices, states):
            action = self.agent.select_action(state, True)
            self._apply_action(action, price_window)

    def _apply_action(self, action, price_window):
        price = price_window[-1]
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
        self.portfolio.append(current_portfolio_value)

    def plot(self):
        plt.plot(self.portfolio)
        plt.show()

    def print(self):
        returns = self.portfolio[-1] / self.portfolio[0] - 1.
        print('return = {}'.format(returns))
