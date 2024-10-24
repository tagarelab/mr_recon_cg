"""
   Name: load_raw_data.py
   Purpose:
   Created on: 5/21/2024
   Created by: Heng Sun
   Additional Notes: 
"""

# import TNMR_dealer as td
from denoise import editer

# Change to the actual data location
loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\241021_YongHyun_scan_separatelysaved\\"

interleave = 4  # Change to the actual interleave number
N_ch = 4  # Change to the actual number of channels
rep = 1  # Change to the actual number of separate .tnt files
seg = 1  # Change to the actual number of segments in one .tnt files that will be saved in separate rows, usually 1
sig_ch_name = 'ch2'  # Change to the actual signal channel name
emi_ch_name = ['ch1', 'ch3', 'ch4']  # Change to the actual EMI channel names
editer.editer_process_all_files(loc, interleave, N_ch, rep, seg, sig_ch_name, emi_ch_name,
                                new_file_prefix='')
