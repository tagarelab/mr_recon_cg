"""
   Name: expr_log_241021.py
   Purpose:
   Created on: 10/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import mr_io as mr_io
from denoise import editer as edi

data_mat = mr_io.load_single_mat(name='test_256avg',
                                 path='C:\\Yale\\MRRC\\mr_recon_cg\\data\\241017_YongHyun_6min_scan\\EDITER\\')
raw = data_mat['raw_sig']
editer = data_mat['editer_corr']
comb = mr_io.load_single_mat(name='test_256avg_comb_10212024', path='sim_output/')['comb_sig_all'][:]
editer_comb = mr_io.load_single_mat(name='test_256avg_editer_comb_10212024', path='sim_output/')['comb_sig_all'][:]

datafft = comb
datanoise_fft_list = data_mat['raw_emi']
comb_editer = edi.editer_process_2D(datafft, datanoise_fft_list)

dict = {'raw': raw, 'editer': editer, 'comb': comb, 'editer_comb': editer_comb, 'comb_editer': comb_editer}
mr_io.save_dict(dict, name='all_combos_comb_editer_10212024', path='sim_output/')
