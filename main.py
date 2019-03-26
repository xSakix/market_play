import scipy
import numpy as np
from generator import generator
import matplotlib.pyplot as plt
from agent.agent import Agent
from market_env.market_env import Actions


def main():
    agent = Agent()
    agent.train()


if __name__ == "__main__":
    main()
