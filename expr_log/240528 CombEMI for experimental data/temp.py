"""
   Name: temp.py
   Purpose:
   Created on: 5/28/2024
   Created by: Heng Sun
   Additional Notes: 
"""
import numpy as np
import comb_test_240528 as cs
import visualization as vis
import matplotlib.pyplot as plt

sn = np.array([[10, 24601, 0]]).T
length = 14000
dt = 1e-5
emi = cs.gen_sn(sn, length, dt, bw=0)
vis.complex(emi, name='EMI', rect=True)
vis.freq_plot(emi, dt=dt, name='EMI')

emi_kspace = np.fft.fft(np.fft.fftshift(emi))

emi_windowed = emi_kspace.reshape((200, 70))
emi_windowed = emi_windowed[0:70, :]

emi_img = np.reshape(emi_windowed, (70, 70))
plt.figure()
plt.imshow(np.abs(emi_img))
plt.show()
