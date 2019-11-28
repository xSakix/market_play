from sklearn.preprocessing import MinMaxScaler

from generator.generator import BrownianGenerator
from transformer.transformer import SlidingWindow, MinMaxTransform
from collections import deque
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


class Actions(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    NO_ACTION = -1


class MarketEnv:

    def __init__(self, num_samples=1000, window=30):
        self.eps = np.finfo(np.float32).eps.item()
        self.samples = BrownianGenerator(num_samples).generate()
        print('MIN:', min(self.samples))
        print('MAX:', max(self.samples))
        print('MEDIAN:', np.median(self.samples))
        self.window = window
        # self.samples = MinMaxTransform().transform(data)
        self.prices = SlidingWindow(window).transform(self.samples)
        self.returns = np.log(self.prices[:, 1::]) - np.log(self.prices[:, :-(window - 1)])
        self.queue_prices = deque(self.prices)
        self.queue_states = deque(self.returns)
        self.shares = 0
        self.cash = 100000.
        self.investment = self.cash

    def reset(self):
        self.shares = 0
        self.cash = 100000.
        self.queue_prices = deque(self.prices)
        self.queue_states = deque(self.returns)
        print('Env reset - shared:%d - cash: %.2f - num of data: %d' % (self.shares, self.cash, len(self.queue_prices)))
        return self.queue_states.popleft(), self.queue_prices.popleft(), 0

    def step(self, action, state, price_window):
        # 0 - Hold, 1 - Buy, 2-Sell
        price = price_window[-1]


        if len(self.queue_prices) == 0:
            new_state = state
            new_price_window = price_window
        else:
            new_price_window = self.queue_prices.popleft()
            new_state = self.queue_states.popleft()
            # for a data array of prices with window length
            # Agents buy for last price known
            # But the evaluation needs to be on the next price, which shows
            # how good the decision was...
            # Remark: maybe a better solution for reward would be
            # to compute returns for all prices afterwards
            # and avg the reward?
            # Remark 2: maybe only for the window?
            # reward_price = new_state[0]
            # price_window = new_state

        if action == Actions.SELL.value and self.shares > 0:
            self.cash = self.shares * price
            self.shares = 0
        elif action == Actions.SELL.value:
            action = Actions.NO_ACTION.value

        if action == Actions.BUY.value and self.cash > price:
            part = int(self.cash / price)
            self.shares = self.shares + part
            self.cash = self.cash - part * price
        elif action == Actions.BUY.value:
            action = Actions.NO_ACTION.value

        if action == Actions.HOLD.value:
            pass

        if action == Actions.NO_ACTION.value:
            return new_state, new_price_window, 0, action

        portfolio_value = new_price_window * self.shares + self.cash
        # returns = (portfolio_value / self.investment) - np.ones(len(price_window))
        result = (portfolio_value / self.investment) - 1.
        # return new_state, ((returns - returns.mean()) / (returns.std() + self.eps)).sum()

        # p_last = self.samples[-1] * self.shares + self.cash
        # result = np.power(portfolio_value[0] / portfolio_value, (1. / self.window)) - 1.
        # print(str(portfolio_value[0])+', '+str(portfolio_value[-1]))
        # result = np.power(portfolio_value[0] / portfolio_value[-1], (1. / self.window)) - 1.
        # result = np.log(portfolio_value[1::]) - np.log(portfolio_value[:-1])
        # if self.shares > 0:
        #     print(portfolio_value, ' --> ', result)

        return new_state, new_price_window, result.mean(), action

    def __len__(self):
        return len(self.prices)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.samples)
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = MarketEnv()
    # env.plot()
    # print(env.samples[:30])
    # print(env.queue.popleft())
    # print(len(env))
    # for _ in range(len(env)):
    #     print(env.queue.popleft())
    # print(80 * '-')
    # print(env.states[-1])
    # print("%.8f"%env.samples[-1])
    # s = env.queue_states.popleft()
    # print(s[-1])
    # s, r = env.step(Actions.BUY.value, s)
    # print(s, r)
    plt.plot(env.returns)
    plt.show()
    print(env.prices[0])
    a = np.log(env.prices[0, 1::]) - np.log(env.prices[0, :-29])
    print(a)
    print(len(a))
    print(len(env.prices[0]))
