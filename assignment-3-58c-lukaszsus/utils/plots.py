import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def params_in_last_n_samples(x, window_size):
    means = np.zeros(shape=len(x) - window_size)
    stds = np.zeros(shape=len(x) - window_size)
    for i in range(means.shape[0]):
        means[i] = np.mean(x[i:i + window_size])
        stds[i] = np.std(x[i:i + window_size])
    return means, stds


def plot_data(df, window_size, vlines: list = None, rescale_ylim=False):
    means, stds = params_in_last_n_samples(df.x, window_size)

    fig = plt.figure(figsize=(20, 15))
    ax1 = plt.subplot(3, 1, 1)
    ax1 = sns.lineplot(x=df.index, y=df.x)
    ax1.set_title("data")
    ax2 = plt.subplot(3, 1, 2)
    ax2 = sns.lineplot(x=range(len(means)), y=means)
    ax2.set_title("mean")
    if rescale_ylim:
        ax2.set_ylim(0, 2.0)
    if vlines is not None:
        ax2.vlines(vlines, ymin=np.min(means), ymax=np.max(means))
    ax3 = plt.subplot(3, 1, 3)
    ax3 = sns.lineplot(x=range(len(stds)), y=stds)
    ax3.set_title("std")
    if rescale_ylim:
        ax3.set_ylim(0, 1.0)
    if vlines is not None:
        ax3.vlines(vlines, ymin=np.min(stds), ymax=np.max(stds))