"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td

td.scan_2_mat(loc="C:\\Yale\\MRRC\\mr_recon_cg\\data\\240528 RO_Grad_w_Polarization\\WithRO\\", interleave=1,
              N_ch=1,
              rep=16, name="Scan#", seg=1)
