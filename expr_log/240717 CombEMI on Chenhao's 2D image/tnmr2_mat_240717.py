"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td

loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\240715_Chenhao_2D_image\\"
td.scan_2_mat(loc=loc,
                  interleave=1,
                  N_ch=1,
                  rep=65, name="ThirdTryPE#", seg=1, confirm=False)