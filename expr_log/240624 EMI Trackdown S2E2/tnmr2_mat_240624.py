"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import TNMR_dealer as td

# step = "Step_4_Console_Preamp_Coil"
# step = "Step_6_Console_Preamp_TRSwitch_Tx_Coil"
# step = "Step_7_Coil_Everything"
# step = "Step_8_Coil_Everything_Pol"
# step = "Step_9_Coil_Everything_Pol_GA"
# step = "Step_10_Coil_Everything_Pol_GA_Phantom"
step = "Step_11_Coil_Everything_Pol_GAPulsing_Phantom"

loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\240624 EMI_hunt_exp_2\\" + step + "\\"

file = ["_withPol_Blank", "_withPol_Phant", "_withPol_Phant_16avg"]
# file = ["","_Blank","_Blank_PAon_nopulsing","_Blank_PAon_pulsing","_PAon_nopulsing","_PAon_pulsing"]

for f in file:
    td.scan_2_mat(loc=loc,
                  interleave=4,
                  N_ch=4,
                  rep=1, name="PolarAcq_EMI_3axis" + f, seg=1, confirm=False, new_name=step + f)
