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

__all__ = ['save_dict', 'save_single_mat', 'load_single_mat', 'load_list_mat']


def read_b1(path, intrp_x, intrp_y, intrp_z, FOV, scale=1):
    # Read the data
    B1_data = pd.read_csv(path, header=None)

    # Extract coordinates and magnetic field components
    B1_X_coord = B1_data.iloc[:, 0].values
    B1_Y_coord = B1_data.iloc[:, 1].values
    B1_Z_coord = B1_data.iloc[:, 2].values
    X_data = B1_data.iloc[:, 3].values
    Y_data = B1_data.iloc[:, 4].values
    Z_data = B1_data.iloc[:, 5].values

    # Calculate the magnitude of the magnetic field and apply a scaling factor
    nubo_b1 = np.sqrt(X_data ** 2 + Y_data ** 2 + Z_data ** 2)
    nubo_b1 = scale * nubo_b1

    # Create a 3D grid for the magnetic field data
    x_M = np.linspace(B1_X_coord.min(), B1_X_coord.max(), 11)
    y_M = np.linspace(B1_Y_coord.min(), B1_Y_coord.max(), 11)
    z_M = np.linspace(B1_Z_coord.min(), B1_Z_coord.max(), 11)
    nubo_b1_mesh, _, _, _ = algebra.vec2mesh(nubo_b1, B1_X_coord, B1_Y_coord, B1_Z_coord, 11, 11, 11)

    nubo_B1_intrp, b1_X_intrp, b1_Y_intrp, b1_Z_intrp = algb.interp_3dmat(nubo_b1_mesh, x_M, y_M, z_M, intrp_x,
                                                                          intrp_y, intrp_z)

    b1_X_intrp = b1_X_intrp / 200 * FOV
    b1_Y_intrp = b1_Y_intrp / 200 * FOV
    b1_Z_intrp = b1_Z_intrp / 200 * FOV

    return nubo_B1_intrp, b1_X_intrp, b1_Y_intrp, b1_Z_intrp


def read_nubo_b0(path, intrp_x, intrp_y, intrp_z, FOV=0.4, scale=2.104):
    # TODO: generalize this function
    # Read the data
    B0_data = pd.read_csv(path, header=None)

    # Extract coordinates and magnetic field components
    B0_X_coord = B0_data.iloc[:, 0].values
    B0_Y_coord = B0_data.iloc[:, 1].values
    B0_Z_coord = B0_data.iloc[:, 2].values
    X_data = B0_data.iloc[:, 3].values
    Y_data = B0_data.iloc[:, 4].values
    Z_data = B0_data.iloc[:, 5].values

    # Calculate the magnitude of the magnetic field and apply a scaling factor
    nubo_b0 = np.sqrt(X_data ** 2 + Y_data ** 2 + Z_data ** 2)
    nubo_b0 = scale * nubo_b0

    # Create a 3D grid for the magnetic field data
    x_M = np.linspace(B0_X_coord.min(), B0_X_coord.max(), 11)
    y_M = np.linspace(B0_Y_coord.min(), B0_Y_coord.max(), 11)
    z_M = np.linspace(B0_Z_coord.min(), B0_Z_coord.max(), 11)
    nubo_b0_mesh, _, _, _ = algebra.vec2mesh(nubo_b0, B0_X_coord, B0_Y_coord, B0_Z_coord, 11, 11, 11)

    nubo_B0_intrp, b0_X_intrp, b0_Y_intrp, b0_Z_intrp = algb.interp_3dmat(nubo_b0_mesh, x_M, y_M, z_M, intrp_x,
                                                                          intrp_y, intrp_z)

    b0_X_intrp = b0_X_intrp / 200 * FOV
    b0_Y_intrp = b0_Y_intrp / 200 * FOV
    b0_Z_intrp = b0_Z_intrp / 200 * FOV

    return nubo_B0_intrp, b0_X_intrp, b0_Y_intrp, b0_Z_intrp


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
    file_name = f'{name}.mat'
    if path is not None:
        file_name = path + file_name
    mdic = sio.loadmat(file_name)

    if disp_msg:
        print('File ' + name + ' loaded')
        print(sorted(mdic.keys()))
    return mdic
