import scipy
from scipy.stats import norm

import numpy as np


class Generator:
    def generate(self):
        pass


class BrownianGenerator(Generator):
    def __init__(self, sample_size):
        self.x0 = 0.
        self.n = sample_size
        self.dt = np.random.uniform(0., 2.)
        self.delta = np.random.uniform(0., 1.)

    def generate(self):
        x0 = np.asarray(self.x0)
        r = norm.rvs(size=x0.shape + (self.n,), scale=self.delta * np.sqrt(self.dt))
        out = np.empty(r.shape)
        np.cumsum(r, axis=-1, out=out)
        out += np.expand_dims(x0, axis=-1)
        return out
