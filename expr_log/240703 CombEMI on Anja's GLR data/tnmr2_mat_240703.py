"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td

loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\240703_Anja_GLR_Data\\"

file = ["GLR_0_rftime80us_90ampl_3.6_200pts_full", "GLR_80_rftime80us_90ampl_3.6_200pts"]

for f in file:
    td.scan_2_mat(loc=loc,
                  interleave=1,
                  N_ch=1,
                  rep=1, name=f, seg=1, confirm=False, new_name=f)
