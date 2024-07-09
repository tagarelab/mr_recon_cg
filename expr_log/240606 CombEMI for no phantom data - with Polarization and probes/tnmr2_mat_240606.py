"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td

td.scan_2_mat(loc="C:\\Yale\\MRRC\\mr_recon_cg\\data\\240606 Polarization_w_3_axis_EMI\\No_phantom\\", interleave=4,
              N_ch=4,
              rep=16, name="Scan_", seg=1)
