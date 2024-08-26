"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td

loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\240820_Anja_64avg_Sinc_Pulse\\"
# td.scan_2_mat(loc=loc,
#               interleave=4,
#               N_ch=4,
#               rep=64, name="30kHz_sinc_iter_", seg=1, confirm=False)
td.scan_2_mat(loc=loc,
              interleave=4,
              N_ch=4,
              rep=1, name="30kHz_att90_16_att180_9.9794_64avg", seg=1, confirm=False)
