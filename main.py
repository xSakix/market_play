import scipy
import numpy as np
from generator import generator
import matplotlib.pyplot as plt
from agent.agent import Agent
from market_env.market_env import Actions
from torch.distributions import Categorical
import torch
from evaluator.marketagentevaluator import MarketAgentEvaluator
import pandas as pd


def main():
    df = pd.read_csv('evaluator/btc_etf_data_adj_close.csv')
    df = df[df.date > '2017-10-01']
    data = df['BTC-EUR'].values
    window=360
    agent = Agent(load_existing=False,window=window)
    agent.train(epochs=5, episodes=5,num_samples=1080)

    evaluator = MarketAgentEvaluator(agent)
    evaluator.evaluate(data,window=window)
    evaluator.plot()
    evaluator.print()


if __name__ == "__main__":
    main()
