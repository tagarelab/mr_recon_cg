"""
   Name: editer.py
   Purpose:
   Created on: 8/20/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import TNMR_dealer as td
import mr_io
import visualization as vis
import os


def main():
    # %% Load data using prompts
    # loc = ""  # Change to the actual data location
    # file_name = ""  # Change to the actual name without .tnt. If multiple files, the name need to end with 1,2,3...
    # td.scan_2_mat(loc=loc, name=file_name, confirm=True)

    # # %% RECOMMENDED: Load data using parameters (comment the block above and uncomment this)
    # loc = ""  # Change to the actual data location
    # file_name = ""  # Change to the actual name without .tnt. If multiple files, the name need to end with 1,2,3...
    # interleave = 4  # Change to the actual interleave number
    # N_ch = 4  # Change to the actual number of channels
    # rep = 1  # Change to the actual number of separate .tnt files
    # seg = 1  # Change to the actual number of segments in one .tnt files that will be saved in separate rows, usually 1
    # td.scan_2_mat(loc=loc,
    #               interleave=interleave,
    #               N_ch=N_ch,
    #               rep=rep, name=file_name, seg=seg, confirm=False)
    #
    # # %% Apply EDITER to one data file
    # data_mat = mr_io.load_single_mat(name=file_name, path=loc)
    # sig_ch_name = 'ch2'  # Change to the actual signal channel name
    # emi_ch_name = ['ch1', 'ch3', 'ch4']  # Change to the actual EMI channel names
    # datafft = data_mat[sig_ch_name]
    # datanoise_fft_list = [data_mat[name] for name in emi_ch_name]
    # editer_corr = editer_process_2D(datafft, datanoise_fft_list)
    #
    # # %% Visualization
    # region = [0, 500]
    # vis.absolute(datafft[region[0]:region[1]], name='Uncorrected')
    # vis.absolute(editer_corr[region[0]:region[1]], name='Corrected with EDITER, Zoom in to %s' % (
    #     region))
    #
    # # %% Save the corrected data
    # data_mat['editer_corr'] = editer_corr
    # mr_io.save_dict(data_mat, name=file_name + '_EDITER', path=loc)

    # %% Apply EDITER to all data files in the folder (Comment out EVERYTHING above this block)
    loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\240902_Anja_Y_phantom\\09_02_2024\\RO_PE\\"
    # Change to the actual data location
    interleave = 4  # Change to the actual interleave number
    N_ch = 4  # Change to the actual number of channels
    rep = 1  # Change to the actual number of separate .tnt files
    seg = 1  # Change to the actual number of segments in one .tnt files that will be saved in separate rows, usually 1
    editer_process_all_files(loc, interleave, N_ch, rep, seg)


def editer_process_all_files(folder_path, interleave, N_ch, rep, seg):
    # List all files in the folder
    files = os.listdir(folder_path)
    output_path = os.path.join(folder_path, "EDITER\\")
    # Check if the output folder already exists
    if not os.path.exists(output_path):
        # Create the output folder
        os.makedirs(output_path)

    # Sort the files to ensure consistent order
    files.sort()

    # Loop through each file and perform the desired operation
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            # Perform the desired operation on the file
            print(f"Processing file: {file_path}")
            if file.endswith(".tnt"):
                file = file[:-4]
            data_mat = td.scan_2_mat(loc=folder_path, name=file, interleave=interleave, N_ch=N_ch, rep=rep, seg=seg,
                                     confirm=False, save=False, return_data=True, disp_msg=False)
            sig_ch_name = 'ch2'  # Change to the actual signal channel name
            emi_ch_name = ['ch1', 'ch3', 'ch4']  # Change to the actual EMI channel names
            datafft = data_mat[sig_ch_name]
            datanoise_fft_list = [data_mat[name] for name in emi_ch_name]
            editer_corr = editer_process_2D(datafft, datanoise_fft_list)
            mr_io.save_single_mat(editer_corr, name=file + '_EDITER', path=output_path, disp_msg=False)


def editer_process_2D(datafft, datanoise_fft_list):
    """
    Translated from MATLAB to Python with ChatGPT based on the original EDITER code by Sai Abitha Srinivas.
    Edited & tested by the author.
    Process 2D brain slice data using a broadband EMI algorithm.

    Parameters:
        datafft: 2D numpy array, k space data.
        datanoise_fft_list: List of 2D numpy arrays, k space noise data from each coil.
        Nc: int, number of channels.

    Returns:
        corr_img_opt_toep: 2D numpy array, processed image after correction.
    """

    # Number of EMI coils
    Nc = len(datanoise_fft_list)

    # Check Data size
    if datafft.ndim == 1:
        datafft = np.expand_dims(datafft, axis=1)
    if datanoise_fft_list[0].ndim == 1:
        datanoise_fft_list = [np.expand_dims(datanoise_fft, axis=1) for datanoise_fft in datanoise_fft_list]

    # Image size
    ncol, nlin = datafft.shape

    # Initial pass using single PE line (Nw=1)
    ksz_col, ksz_lin = 0, 0

    # Kernels across PE lines
    kern_pe = np.zeros((Nc * (2 * ksz_col + 1) * (2 * ksz_lin + 1), nlin), dtype='complex')

    for clin in range(nlin):
        noise_mat = []

        pe_rng = [clin]

        padded_dfs = [np.pad(datanoise_fft[:, pe_rng], ((ksz_col, ksz_col), (ksz_lin, ksz_lin)), mode='constant') for
                      datanoise_fft in datanoise_fft_list]

        for col_shift in range(-ksz_col, ksz_col + 1):
            for lin_shift in range(-ksz_lin, ksz_lin + 1):
                for padded_df in padded_dfs:
                    dftmp = np.roll(np.roll(padded_df, col_shift, axis=0), lin_shift, axis=1)
                    if ksz_col > 0:
                        dftmp = dftmp[ksz_col:-ksz_col, :]
                    if ksz_lin > 0:
                        dftmp = dftmp[:, ksz_lin:-ksz_lin]
                    noise_mat.append(dftmp)

        gmat = np.reshape(np.stack(noise_mat, axis=-1), (-1, len(noise_mat)))

        init_mat_sub = datafft[:, pe_rng]
        kern = np.linalg.lstsq(gmat, init_mat_sub.flatten(), rcond=None)[0]
        kern_pe[:, clin] = kern

    # Normalize kernels
    kern_pe_normalized = kern_pe / np.linalg.norm(kern_pe, axis=0, keepdims=True)
    kcor = np.abs(np.dot(kern_pe_normalized.T, kern_pe_normalized))

    # Threshold
    default_thresh = 5e-1
    kcor_thresh = kcor > default_thresh
    while np.sum(kcor_thresh) == 0:
        default_thresh = default_thresh / 5
        kcor_thresh = kcor > default_thresh

    # Start with the full set of lines
    aval_lins = list(range(nlin))

    # Window stack
    win_stack = []

    while aval_lins:
        clin = aval_lins[0]
        pe_rng = list(range(clin, clin + np.max(np.where(kcor_thresh[clin, clin:])) + 1))
        win_stack.append(pe_rng)
        aval_lins = sorted(set(aval_lins) - set(pe_rng))

    # Final processing with the selected lines
    ksz_col, ksz_lin = 7, 0
    gksp = np.zeros((ncol, nlin), dtype='complex')

    for pe_rng in win_stack:
        noise_mat = []

        padded_dfs = [np.pad(datanoise_fft[:, pe_rng], ((ksz_col, ksz_col), (ksz_lin, ksz_lin)), mode='constant') for
                      datanoise_fft in datanoise_fft_list]

        for col_shift in range(-ksz_col, ksz_col + 1):
            for lin_shift in range(-ksz_lin, ksz_lin + 1):
                for padded_df in padded_dfs:
                    dftmp = np.roll(np.roll(padded_df, col_shift, axis=0), lin_shift, axis=1)
                    if ksz_col > 0:
                        dftmp = dftmp[ksz_col:-ksz_col, :]
                    if ksz_lin > 0:
                        dftmp = dftmp[:, ksz_lin:-ksz_lin]
                    noise_mat.append(dftmp)

        gmat = np.reshape(np.stack(noise_mat, axis=-1), (-1, len(noise_mat)))

        init_mat_sub = datafft[:, pe_rng]
        kern = np.linalg.lstsq(gmat, init_mat_sub.flatten(), rcond=None)[0]

        # Put the solution back
        tosub = np.reshape(np.dot(gmat, kern), (ncol, len(pe_rng)))
        gksp[:, pe_rng] = init_mat_sub - tosub

    return gksp


if __name__ == "__main__":
    main()
