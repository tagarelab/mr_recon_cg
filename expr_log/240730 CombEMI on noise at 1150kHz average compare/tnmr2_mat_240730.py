"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td

loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\240724_EMI_Analysis_1150kHz\\avg_compare\\"

file = ["Noisy_Sig_"]

for f in file:
    td.scan_2_mat(loc=loc,
                  interleave=1,
                  N_ch=1,
                  rep=64, name=f, seg=1, confirm=False, new_name=f)
