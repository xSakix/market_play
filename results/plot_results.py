import matplotlib.pyplot as plt
import numpy as np


def plot_results(dir="1/"):
    training_loss = np.load(dir + "training_loss.npy")
    training_rewards = np.load(dir + "training_rewards.npy")
    portfolios = np.load(dir + "portfolios.npy")

    plt.plot(training_loss, training_rewards, 'o')
    plt.xlabel("loss")
    plt.ylabel("reward")
    plt.show()

    plt.plot(training_loss, portfolios, 'o')
    plt.xlabel("loss")
    plt.ylabel("portfolio")
    plt.show()

    plt.plot(1. - training_loss)
    plt.title("training loss")
    plt.show()

    plt.plot(training_rewards)
    plt.title("training rewards")
    plt.show()


    plt.plot(portfolios)
    plt.title("portfolio")
    plt.show()



if __name__ == "__main__":
    plot_results("non_negative_loss/7.1000/")
