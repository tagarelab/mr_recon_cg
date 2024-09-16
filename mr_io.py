"""
   Name: mr_io.py
   Purpose: Input/Output functions for the project
   Created on: 2/22/2024
   Created by: Heng Sun
   Additional Notes: 
"""

from scipy import io as sio
import time
import pandas as pd
import numpy as np

import algebra
import algebra as algb
import visualization as vis

__all__ = ['save_dict', 'save_single_mat', 'load_single_mat', 'load_list_mat']



def read_nubo_b0(path, intrp_pts, scale_b0=2.104):
    B0_data = pd.read_csv(path, header=None)
    # Extract coordinates and magnetic field components
    X_axis_coord = B0_data.iloc[:, 0].values * 2  # mm
    Y_axis_coord = B0_data.iloc[:, 1].values * 2  # mm
    Z_axis_coord = B0_data.iloc[:, 2].values * 2  # mm

    nubo_b0 = B0_data.iloc[:, 3].values
    nubo_b0 = scale_b0 * nubo_b0
    nubo_b0_mesh, x_M, y_M, z_M = algb.vec2mesh(nubo_b0, X_axis_coord, Y_axis_coord, Z_axis_coord, 11, 11, 11)
    nubo_b0_mesh = nubo_b0_mesh.T
    B0_x = algb.interp_by_pts(nubo_b0_mesh, x_M, y_M, z_M, intrp_pts, method='linear')

    nubo_b0 = B0_data.iloc[:, 4].values
    nubo_b0 = scale_b0 * nubo_b0
    nubo_b0_mesh, x_M, y_M, z_M = algb.vec2mesh(nubo_b0, X_axis_coord, Y_axis_coord, Z_axis_coord, 11, 11, 11)
    nubo_b0_mesh = nubo_b0_mesh.T
    B0_y = algb.interp_by_pts(nubo_b0_mesh, x_M, y_M, z_M, intrp_pts, method='linear')

    nubo_b0 = B0_data.iloc[:, 5].values
    nubo_b0 = scale_b0 * nubo_b0
    nubo_b0_mesh, x_M, y_M, z_M = algb.vec2mesh(nubo_b0, X_axis_coord, Y_axis_coord, Z_axis_coord, 11, 11, 11)
    nubo_b0_mesh = nubo_b0_mesh.T
    B0_z = algb.interp_by_pts(nubo_b0_mesh, x_M, y_M, z_M, intrp_pts, method='linear')

    return np.array([B0_x, B0_y, B0_z])


def read_b0(path, intrp_x, intrp_y, intrp_z, scale, FOV):
    # TODO: implement this function
    return 0


def read_csv(path, header=None):
    return pd.read_csv(path, header=header)


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
    if not name.endswith('.mat'):
        file_name = f'{name}.mat'
    else:
        file_name = name
        name = name[:-4]

    if path is not None:
        file_name = path + file_name

    mdic = sio.loadmat(file_name)

    if disp_msg:
        print('File ' + name + ' loaded')
        print(sorted(mdic.keys()))
    return mdic
