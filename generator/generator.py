import numpy as np
from scipy.stats import norm


class Generator:
    def generate(self):
        pass


class BrownianGenerator(Generator):
    def __init__(self, sample_size, x0=np.random.randint(10, 100)):
        self.x0 = x0
        self.n = sample_size
        self.dt = np.random.uniform(0., 2.)
        self.delta = np.random.uniform(0., 1.)

    def generate(self):
        x0 = np.asarray(self.x0)
        r = norm.rvs(size=x0.shape + (self.n,), scale=self.delta * np.sqrt(self.dt))
        out = np.empty(r.shape)
        np.cumsum(r, axis=-1, out=out)
        out += np.expand_dims(x0, axis=-1)
        out = np.abs(out)
        return out


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    for _ in range(10):
        br = BrownianGenerator(63000)
        plt.plot(br.generate())
        plt.show()
