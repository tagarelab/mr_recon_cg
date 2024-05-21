"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td
import data

td.scan_2_mat(loc="C:\\Yale\\MRRC\\mr_recon_cg\\data\\240517 Chenhao_2D_grad_encoded\\2D scan data\\", interleave=1,
              N_ch=1,
              rep=33, name="FirstTryPE#", seg=1)
