import numpy as np
from matplotlib import pyplot as plt

def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

def plot_episodes(score):
    score = np.array(score)
    avg_score = running_mean(score, 50)
    plt.figure(figsize=(10,7))
    plt.ylabel("Episode Duration",fontsize=22)
    plt.xlabel("Training Epochs",fontsize=22)
    plt.plot(avg_score, color='green')
    plt.show()