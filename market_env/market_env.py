from generator import generator
from transformer.transformer import SlidingWindow


class MarketEnv:
    def __init__(self):
        self.samples = generator.BrownianGenerator(1000).generate()
        self.states = SlidingWindow(30).transform(self.samples)
        self.shares = 0
        self.cash = 100000.

    def reset(self):
        self.shares = 0
        self.cash = 100000.
        return self.states

    def step(self, action, state):
        # 0 - Hold, 1 - Buy, 2-Sell

        pass
