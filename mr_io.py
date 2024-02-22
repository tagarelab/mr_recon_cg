"""
   Name: mr_io.py
   Purpose: Input/Output functions for the project
   Created on: 2/22/2024
   Created by: Heng Sun
   Additional Notes: 
"""

from scipy import io as sio
import time

__all__ = ['save_dict', 'save_single_mat', 'load_single_mat', 'load_list_mat']


def save_dict(mdic, name, path=None, date=False, disp_msg=True):
    file_name = name

    if date:
        file_name = file_name + '_' + time.strftime("%m%d%Y")

    if path is not None:
        file_name = path + file_name

    sio.savemat(file_name + '.mat', mdic)

    if disp_msg:
        print('File saved as: ' + file_name)
        print(sorted(mdic.keys()))


def save_single_mat(data, name, path=None, date=False, disp_msg=True):
    mdic = {"data": data}
    save_dict(mdic, name, path, date, disp_msg)


def load_list_mat(name, num_files, path=None, disp_msg=True):
    """

    :param name: data name, end with natual numbers for # of position encodings
    :param num_files: number of files to load
    :param path: directory name for data
    :param disp_msg: whether to display a message when finished
    :return: a list of dictionaries
    """
    data_list = []
    for ii in range(1, num_files + 1):
        string_name_file = f'{name}{ii}'
        data = load_single_mat(string_name_file, path, disp_msg=False)
        data_list.append(data)

    if disp_msg:
        print(f'{num_files} files {name} loaded')

    return data_list


def load_single_mat(name, path=None, disp_msg=True):
    file_name = f'{name}.mat'
    if path is not None:
        file_name = path + file_name
    mdic = sio.loadmat(file_name)

    if disp_msg:
        print('File ' + name + ' loaded')
        print(sorted(mdic.keys()))
    return mdic
