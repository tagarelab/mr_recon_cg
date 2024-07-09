"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td

td.scan_2_mat(loc="C:\\Yale\\MRRC\\mr_recon_cg\\data\\240610 EMI_w_prepol_on_off\\NoPhantom_WithPolarization\\",
              interleave=4,
              N_ch=4,
              rep=16, name="Scan_", seg=1)
