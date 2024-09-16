"""
   Name: noise_analysis.py
   Purpose: Noise analysis functions
   Created on: 9/9/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np


def power(data):
    """
    Compute the power of the data.
    :param data:
    :return:
    """
    return np.mean(np.abs(data) ** 2)


def snr(data, signal_region=None, noise_region=None, method='std'):
    """
    Compute the signal-to-noise ratio (SNR) of the data.
    This function is drafted by Github Copilot and edited & tested by the author.

    Parameters:
    - data (numpy.ndarray): The data.
    - signal_region (tuple): The region of the signal.
    - noise_region (tuple): The region of the noise.
    - method (str): The method to compute SNR. Default is 'std'.

    Returns:
    - float: The SNR of the data.
    """

    if signal_region is None and noise_region is None:
        print("Signal region is not specified, using the entire data as both signal region and noise region.")
        signal_region = (0, len(data))
        noise_region = (0, len(data))

    if noise_region is None:
        print("Noise region is not specified, using everything except signal region.")
        noise_region = (0, signal_region[0]) + (signal_region[1], len(data))

    if signal_region is None:
        print("Signal region is not specified, using what is not noise region.")
        signal_region = (0, noise_region[0]) + (noise_region[1], len(data))

    if method == 'std':
        signal = data[signal_region[0]:signal_region[1]]
        noise = data[noise_region[0]:noise_region[1]]
        snr = np.mean(signal) / np.std(noise)

    return snr


def abs_snr(data, signal_region=None, noise_region=None, method='std'):
    """
    Compute the signal-to-noise ratio (SNR) of the absolute value of the data.
    :param data:
    :param signal_region:
    :param noise_region:
    :param method:
    :return:
    """
    return snr(np.abs(data), signal_region, noise_region, method)


def freq_domain_snr(time_domain_data, dt, signal_region=None, noise_region=None, signal_freq_range=None,
                    noise_freq_range=None, method='std'):
    """
    Compute the signal-to-noise ratio (SNR) in the frequency domain.
    :param data:
    :param dt:
    :param signal_freq:
    :param noise_freq:
    :param method:
    :return:
    """
    freq_domain_data = np.fft.fftshift(np.fft.fft(time_domain_data))
    freq = np.fft.fftshift(np.fft.fftfreq(len(time_domain_data), dt))
    if signal_region is None:
        if signal_freq_range is not None:
            signal_region = (np.argmax(freq > signal_freq_range[0]), np.argmax(freq > signal_freq_range[1]))

    if noise_region is None:
        if noise_freq_range is not None:
            noise_region = (np.argmax(freq > noise_freq_range[0]), np.argmax(freq > noise_freq_range[1]))
    return snr(freq_domain_data, signal_region, noise_region, method)
