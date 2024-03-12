"""
   Name: masks.py
   Purpose: Generate masks
   Created on: 3/7/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import matplotlib.pyplot as plt


def mask2matrix(data, mask, x, y, z):
    """
    Convert masked data to a 3D matrix.
    :param data: masked data, 1D array
    :param mask: mask, 3D boolean array
    :param x: x-axis
    :param y: y-axis
    :param z: z-axis
    :return: data in 3D matrix
    """
    data_matrix = np.zeros((len(x), len(y), len(z)), dtype=complex)
    counter = 0
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                if mask[i, j, k]:
                    data_matrix[i, j, k] = data[counter]
                    counter += 1
    return data_matrix


def gen_breast_mask(x, y, z, R=0.06, height=0.100):
    """
    Generate a breast mask. The breast is modeled as a half ellipsoid.
    This function is adapted from code from Yonghyun and edited & tested by the author.
    :param x:
    :param y:
    :param z:
    :param R:
    :param height:
    :return:
    """
    mask = np.zeros((len(x), len(y), len(z)), dtype=bool)
    for i in range(len(x)):
        if x[i] < 0:
            r = np.sqrt(R ** 2 * (1 - (z[i] / height) ** 2)) - 0.005
            for j in range(len(y)):
                for k in range(len(z)):
                    if x[j] ** 2 + z[k] ** 2 < r ** 2:
                        mask[i, j, k] = True

    return mask
