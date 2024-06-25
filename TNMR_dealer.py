# File name: TNMR_dealer.py
# Purpose: do what TNMR can do and can not do with the tnt files
# Author: Heng Sun
# Email: heng.sun@yale.edu
# Date: 6/2/2022

import numpy as np
from pytnt import TNTfile
import mr_io

__all__ = ['read_tnt_1d', 'sort_channels']


def scan_2_mat(loc=None, interleave=None, N_ch=None, rep=None, name=None, seg=None, confirm=True, new_name=None):
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

    data = sort_channels(tnt_data, interleave)
    ch_len = get_ch_len(data, N_ch)
    seg_len = int(ch_len / seg)

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
            data_2d = np.transpose(data.reshape(int(N_ch * seg), seg_len))
            for j in range(N_ch):
                data_mat[:, :, j] = data_2d[:, j * seg:(j + 1) * seg]
        else:
            for i in range(rep):
                tnt_data = TNTfile(loc + name + str(i + 1) + '.tnt')
                data = sort_channels(tnt_data, interleave)
                data_2d = np.transpose(data.reshape(int(N_ch * seg), seg_len))
                for j in range(N_ch):
                    data_mat[:, i * seg:(i + 1) * seg, j] = data_2d[:, j * seg:(j + 1) * seg]

        mdic = {}
        for j in range(N_ch):
            mdic['ch' + str(j + 1)] = data_mat[:, :, j]

        if new_name is None:
            new_name = name
        mr_io.save_dict(mdic, name=new_name, path=loc, date=False)

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


def sort_channels(tnt, interleave=4):
    """Sort the channels in tnt file with a given interleave

    Args:
        tnt (tnt): the tnt file with the data.

    Returns:
        data_sorted: sorted memmap array.

    """
    data_raw = read_tnt_1d(tnt)
    data_length = len(data_raw)
    data_sorted = np.array([])

    for i in range(interleave):
        data_sorted = np.concatenate((data_sorted, data_raw[i:data_length:interleave]))

    return data_sorted
