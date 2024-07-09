"""
   Name: visualization_240422.py
   Purpose:
   Created on: 4/23/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import matplotlib.pyplot as plt

# visualization parameters
short_sig_len = 200  # length of the signal to be visualized


# Short signal
def plot_short(sig, length=short_sig_len, title=None, xlabel=None, ylabel=None):
    """
    Plot a short signal
    :param sig: signal to be plotted
    :param title: title of the plot
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :return: None
    """
    plt.figure()
    plt.plot(abs(sig[:length]))
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()
