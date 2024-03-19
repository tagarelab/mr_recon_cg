"""
   Name: masks.py
   Purpose: Generate masks
   Created on: 3/7/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import matplotlib.pyplot as plt


def apply_mask(data, mask):
    """
    Apply a 3D mask to a 4D data array and output a 2D array.

    :param data: 4D data array
    :param mask: 3D boolean mask
    :return: 2D array with masked data
    """
    # Check if the last three dimensions of data match the dimensions of mask
    assert data.shape[-3:] == mask.shape, "The last three dimensions of data must match the dimensions of mask"

    # Initialize an empty list to store the masked data
    masked_data = []

    # Iterate over the first dimension of data
    for i in range(data.shape[0]):
        # Apply the mask to the 3D data at the current index
        masked_3d_data = data[i][mask]
        # Append the flattened masked data to the list
        masked_data.append(masked_3d_data.flatten())

    # Convert the list to a 2D numpy array
    masked_data = np.array(masked_data)

    return masked_data


def mask2matrix(data, mask, x, y, z):
    """
    Convert masked data to a 3D matrix.
    :param data: masked data, 2D array
    :param mask: mask, 3D boolean array
    :param x: x-axis
    :param y: y-axis
    :param z: z-axis
    :return: data in 3D matrix
    """
    unit_dim = 1
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=-1)
    elif len(data.shape) == 2:
        unit_dim = data.shape[1]

    data_matrix = np.zeros((len(x), len(y), len(z), unit_dim), dtype=complex)
    counter = 0
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                if mask[i, j, k]:
                    for d in range(unit_dim):
                        data_matrix[i, j, k, d] = data[counter, d]
                    counter += 1

    if unit_dim == 1:
        data_matrix = np.squeeze(data_matrix, axis=3)

    return data_matrix


def gen_breast_mask(x, y, z, loc=[0, 0, 0], R=0.06, height=0.100):
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


def gen_sphere(X_axis, Y_axis, Z_axis, loc, rad):
    """
    Create a sphere phantom/mask.

    Parameters:
    - loc (numpy.ndarray): The location of the sphere.
    - rad (float): The radius of the sphere.
    - dim (tuple): The dimensions of the phantom.

    Returns:
    - numpy.ndarray: The sphere phantom.
    """

    X, Y, Z = np.meshgrid(X_axis, Y_axis, Z_axis)

    return (X - loc[0]) ** 2 + (Y - loc[1]) ** 2 + (Z - loc[2]) ** 2 < rad ** 2
