"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td

loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\240726_Chenhao_2D_Y_image\\Fri26_toSat27\\"
td.scan_2_mat(loc=loc,
              interleave=4,
              N_ch=4,
              rep=65, name="SeventhTryAve#1PE#", seg=1, confirm=False)
