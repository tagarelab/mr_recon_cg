# File name: TNMR_dealer.py
# Purpose: do what TNMR can do and can not do with the tnt files
# Author: Heng Sun
# Email: heng.sun@yale.edu
# Date: 6/2/2022

import numpy as np
from pytnt import TNTfile
import mr_io

__all__ = ['read_tnt_1d', 'sort_channels']


def scan_2_mat(loc=None, interleave=None, N_ch=None, rep=None, name=None, seg=None, confirm=True, new_name=None,
               save=True, return_data=False, disp_msg=True):
    if not save and not return_data:
        raise ValueError('Either save or return_data should be True.')

    # Prompt user input if not provided when function is called
    if loc is None:
        loc = input('What is the directory of your data (end with \\)?\n')
    if interleave is None:
        interleave = int(input('What is the interleave (input int)?\n'))
    if N_ch is None:
        N_ch = int(input('How many channels in total (input int)?\n'))
    if rep is None:
        rep = int(input('How many repetitions are there stored in separate files (input int)?\n'))
    if name is None:
        print('The files needs to be stored in the format starting with xxxx1.tnt')
        name = input('File name (not including 1.tnt for rep>1, not including .tnt for rep = 1): ')
    if seg is None:
        seg = int(input('How many repetitions are there stored in one file (input int)?\n'))

    if rep == 1:
        tnt_data = TNTfile(loc + name + '.tnt')
    else:
        tnt_data = TNTfile(loc + name + '1.tnt')

    # data = np.zeros(tnt_data.shape, dtype='complex')
    data = sort_channels(tnt_data, interleave)
    ch_len = get_ch_len(data, N_ch)
    seg_len = int(ch_len / seg)
    if data.ndim == 2:
        if seg == 1:
            seg = data.shape[1]
        else:
            raise ValueError('Not supported for save data as a different number of segments if the data is 2D.')

    if disp_msg:
        print('Output matrix size per channel:\n repetition axis: {0:d}, data axis: {1:d}\n'.format(rep * seg, seg_len))
    if confirm:
        proceed = input('Proceed to saving (yes/no)?')
        if proceed == 'yes':
            confirm = False
    if not confirm:
        data_mat = np.zeros((seg_len, rep * seg, N_ch), dtype='complex')

        if rep == 1:
            tnt_data = TNTfile(loc + name + '.tnt')
            data = sort_channels(tnt_data, interleave)
            if data.ndim == 1:
                data_2d = np.transpose(data.reshape(int(N_ch * seg), seg_len))
                for j in range(N_ch):
                    data_mat[:, :, j] = data_2d[:, j * seg:(j + 1) * seg]
            elif data.ndim == 2:
                data_2d = data
                for j in range(N_ch):
                    data_mat[:, :, j] = data_2d[j * seg_len:(j + 1) * seg_len, :]
        else:
            for i in range(rep):
                tnt_data = TNTfile(loc + name + str(i + 1) + '.tnt')
                data = sort_channels(tnt_data, interleave)
                if data.ndim == 1:
                    data_2d = np.transpose(data.reshape(int(N_ch * seg), seg_len))
                    for j in range(N_ch):
                        data_mat[:, i * seg:(i + 1) * seg, j] = data_2d[:, j * seg:(j + 1) * seg]
                elif data.ndim == 2:
                    raise RuntimeError('Not implemented for saving 2D data with multiple reps.')

        mdic = {}
        for j in range(N_ch):
            mdic['ch' + str(j + 1)] = data_mat[:, :, j]

        if new_name is None:
            new_name = name

        if save:
            mr_io.save_dict(mdic, name=new_name, path=loc, date=False)

        if return_data:
            return mdic

    else:
        print('Data not saved.')

def get_ch_len(data, N_ch):
    """
    Get the length of each channel
    :param data: the entire ndarray of data
    :param N_ch: number of channels
    :return: length of each channel
    """
    return int(len(data) / N_ch)


def read_tnt_1d(tnt):
    """
    Read the 1-d data in tnt file
    :param tnt: the tnt file with the data
    :return: 1-d memmap array with the data
    """
    return tnt.DATA.reshape(-1)


def read_tnt_squeezed(tnt):
    """
    Read the 1-d data in tnt file and squeeze it
    :param tnt: the tnt file with the data
    :return: 1-d memmap array with the data
    """
    return np.squeeze(tnt.DATA)


# def sort_channels(tnt, interleave=4):
#     """Sort the channels in tnt file with a given interleave
#
#     Args:
#         tnt (tnt): the tnt file with the data.
#
#     Returns:
#         data_sorted: sorted memmap array.
#
#     """
#     data_raw = np.squeeze(tnt.DATA)
#     data_length = data_raw.shape[0]
#     data_sorted = np.array([])
#
#     for i in range(interleave):
#         data_sorted = np.concatenate((data_sorted, data_raw[i:data_length:interleave]))
#
#     return data_sorted


def sort_channels(tnt, interleave=4):
    """Sort the channels in tnt file with a given interleave, processing only the first dimension.

    Args:
        tnt (object): the tnt file object with the data.
        interleave (int, optional): the interleave value to use. Defaults to 4.

    Returns:
        np.ndarray: sorted memmap array.
    """
    data_raw = np.squeeze(tnt.DATA)  # Assuming tnt.DATA is a numpy array
    data_length = data_raw.shape[0]

    # Check if interleave is compatible with data length
    if interleave < 1 or interleave > data_length:
        raise ValueError("Interleave must be between 1 and the length of the data.")

    # Calculate the size of the new sorted array
    sorted_length = (data_length // interleave) * interleave

    # Initialize an empty array with the required shape
    data_sorted_shape = (sorted_length,) + data_raw.shape[1:]
    data_sorted = np.empty(data_sorted_shape, dtype=data_raw.dtype)

    index = 0
    for i in range(interleave):
        selected_data = data_raw[i:data_length:interleave]
        num_elements = selected_data.shape[0]

        data_sorted[index:index + num_elements] = selected_data
        index += num_elements

    return data_sorted