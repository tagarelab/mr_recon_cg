"""
   Name:
   Purpose: generate plots from simulation output
   Created on:
   Created by: Heng Sun
   Additional Notes:
"""

## Packages and Data
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import visualization as vis

data_name = 'Int_Comb_# of Injected EMI_Empty_04292024_1'
data_file = sp.io.loadmat('sim_output/' + data_name + '.mat')

# Load data
param1 = np.squeeze(data_file['param1'])
param2 = np.squeeze(data_file['param2'])
param1_name = data_file['param1_name'][0]
param2_name = data_file['param2_name'][0]
pc_comb = data_file['pc_comb']
rmse_comb_freq = data_file['rmse_comb_freq']
rmse_comb_img = data_file['rmse_comb_img']
rmse_org_freq = data_file['rmse_org_freq']
rmse_org_img = data_file['rmse_org_img']
rmse_pro_freq = data_file['rmse_pro_freq']
rmse_pro_img = data_file['rmse_pro_img']

pc_comb_avg = np.mean(pc_comb, axis=0)
rmse_comb_freq_avg = np.mean(rmse_comb_freq, axis=0)
rmse_comb_img_avg = np.mean(rmse_comb_img, axis=0)
rmse_org_freq_avg = np.mean(rmse_org_freq, axis=0)
rmse_org_img_avg = np.mean(rmse_org_img, axis=0)
rmse_pro_freq_avg = np.mean(rmse_pro_freq, axis=0)
rmse_pro_img_avg = np.mean(rmse_pro_img, axis=0)

pc_comb_std = np.std(pc_comb, axis=0)
rmse_comb_freq_std = np.std(rmse_comb_freq, axis=0)
rmse_comb_img_std = np.std(rmse_comb_img, axis=0)
rmse_org_freq_std = np.std(rmse_org_freq, axis=0)
rmse_org_img_std = np.std(rmse_org_img, axis=0)
rmse_pro_freq_std = np.std(rmse_pro_freq, axis=0)
rmse_pro_img_std = np.std(rmse_pro_img, axis=0)

# Plot RMSE with errorbar
plt.figure()
plt.errorbar(param1, rmse_org_img_avg[:, 0], yerr=rmse_org_img_std[:, 0], label='No Correction')
plt.errorbar(param1, rmse_pro_img_avg[:, 0], yerr=rmse_pro_img_std[:, 0], label='Probe Corrected')
plt.errorbar(param1, rmse_comb_img_avg[:, 0], yerr=rmse_comb_img_std[:, 0], label='Comb Corrected')
plt.xlabel(param1_name)
plt.ylabel('RMSE')
plt.title('RMSE vs ' + param1_name)
plt.legend()
plt.show()

# Plot percentage corrected
plt.figure()
plt.errorbar(param1, pc_comb_avg[:, 0], yerr=pc_comb_std[:, 0])
plt.xlabel(param1_name)
plt.ylabel('% Residue')
plt.title('Comb % Residue vs ' + param1_name)
plt.ylim([0, 100])
plt.show()
