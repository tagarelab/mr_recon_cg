"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td

loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\240613 EMI_hunt_exp_1\\Step7_Console_Preamp_TRSwitch_Tx_PA_Coil\\"

td.scan_2_mat(loc=loc,
              interleave=4,
              N_ch=4,
              rep=4, name="Sig_3AxisEMI_", seg=1)

td.scan_2_mat(loc=loc,
              interleave=1,
              N_ch=1,
              rep=4, name="SigOnly_", seg=1)
