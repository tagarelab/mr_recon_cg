"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td


# step = "Magnet_Off"
# step = "Magnet_On"
step = "Magnet_On_Slice7.25"

loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\240724_EMI_Analysis_1150kHz\\" + step + "\\"

file = ["mag_on_s7.25_"]
# file = ["","_Blank","_Blank_PAon_nopulsing","_Blank_PAon_pulsing","_PAon_nopulsing","_PAon_pulsing"]

for f in file:
    td.scan_2_mat(loc=loc,
                  interleave=1,
                  N_ch=1,
                  rep=4, name="PolarAcq_noPol_1150kHz_no_b0_no_phant_" + f, seg=1, confirm=False, new_name=f)
