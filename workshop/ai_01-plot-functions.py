import numpy as np
import matplotlib.pyplot as plt


def plot_sin(x):
    return np.sin(x)


def plot_exp(x):
    return np.exp(x)


def plot_sigmoid(x):
    return 1/(1+np.exp(-x))


def plot_tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


def plot_ReLU(x):
    return np.maximum(0, x)


def plot_LeaklyReLU(x):
    return np.array([i if i > 0 else 0.01*i for i in x])


def plot(func):
    x = np.arange(-10, 10, 0.1)
    y = func(x)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    plot(plot_sigmoid)
    plot(plot_tanh)
    plot(plot_ReLU)
    plot(plot_LeaklyReLU)
    plot(plot_exp)
    plot(plot_sin)
