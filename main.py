import scipy
import numpy as np
from generator import generator
import matplotlib.pyplot as plt


def main():
    # [plt.plot(gen_samples()) for _ in range(3)]
    # plt.show()
    s = gen_samples()
    window = 5

    print(sliding_window(s, window))


def gen_samples():
    g = generator.BrownianGenerator(100)
    samples = g.generate()
    return samples




if __name__ == "__main__":
    main()
