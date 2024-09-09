"""
   Name: expr_log_240903.py
   Purpose:
   Created on: 9/3/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import matplotlib.pyplot as plt
import visualization as vis


def generate_complex_white_noise(size):
    # Generate real and imaginary parts of the noise
    real_part = np.random.normal(0, 1, size)
    imaginary_part = np.random.normal(0, 1, size)
    # Combine them to form complex noise
    complex_noise = real_part + 1j * imaginary_part
    return complex_noise


def generate_sin_pulse(frequency, length, dt):
    # Compute the time vector
    t = np.arange(-length / 2, length / 2, 1)
    # Generate the sinc pulse
    sin_pulse = np.sin(2 * np.pi * frequency * t)
    return t, sin_pulse


# Parameters
size = 1024  # Number of samples
dt = 1e-4  # Time step

# Generate complex white noise
complex_noise = generate_complex_white_noise(size)

i = 0
complex_noise[i] += 100 * np.exp(1j * np.pi / 2)

i = 1
complex_noise[i] += 1000

complex_noise = complex_noise + 1e10 * generate_sin_pulse(3e3, size, dt)[1]

# i_list = np.random.choice(size, 1, replace=False)
# for i in i_list:
#     complex_noise[i] += 1000
#     # complex_noise[i] += 1000*np.exp(1j+np.random.uniform(-np.pi, np.pi))

# Plot the k space signal
vis.complex(complex_noise, name="Simulated Original k-space")
# Plot the Fourier transform
vis.freq_plot(complex_noise, dt, name="Frequency Domain of Original k-space")
