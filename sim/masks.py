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


# def mask2matrix(data, mask, matrix_shape, dtype=None):
#     """
#     Convert masked data to a 3D matrix.
#     :param data: masked data, 2D array
#     :param mask: mask, 3D boolean array
#     :param x: x-axis
#     :param y: y-axis
#     :param z: z-axis
#     :return: data in 3D matrix
#     """
#     if dtype is None:
#         dtype = data.dtype
#
#     unit_dim = 1
#     if len(data.shape) == 1:
#         data = np.expand_dims(data, axis=0)
#     elif len(data.shape) == 2:
#         unit_dim = data.shape[0]
#
#     data_matrix = np.zeros((unit_dim, matrix_shape[0],matrix_shape[1], matrix_shape[2]), dtype=dtype)
#     for d in range(unit_dim):
#         counter = 0
#         for i in range(matrix_shape[0]):
#             for j in range(matrix_shape[1]):
#                 for k in range(matrix_shape[2]):
#                     if mask[i, j, k]:
#                         data_matrix[d, i, j, k] = data[d, counter]
#                         counter += 1
#
#     if unit_dim == 1:
#         data_matrix = np.squeeze(data_matrix, axis=0)
#
#     return data_matrix


import numpy as np


def mask2matrix(data, mask, matrix_shape, dtype=None):
    """
    Convert masked data to an nD matrix.
    :param data: masked data, 2D array
    :param mask: mask, nD boolean array
    :param matrix_shape: shape of the nD matrix
    :param dtype: desired data type of the matrix, optional
    :return: data in an nD matrix
    """
    if dtype is None:
        dtype = data.dtype

    unit_dim = 1
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)
    elif len(data.shape) == 2:
        unit_dim = data.shape[0]

    mask_flattened_size = np.sum(mask)
    data_flattened_size = data.shape[1]

    if mask_flattened_size != data_flattened_size:
        raise ValueError(
            f"Mismatch between flattened mask size ({mask_flattened_size}) and data size ({data_flattened_size})")

    # Initialize a matrix filled with zeros of the target shape and specified dtype
    data_matrix = np.zeros((unit_dim, *matrix_shape), dtype=dtype)

    def fill_matrix(data, data_matrix, mask, indices, unit, data_counter):
        if len(indices) == len(matrix_shape):
            if mask[tuple(indices)]:
                if data_counter[0] >= data.shape[1]:
                    raise IndexError(f"Attempt to access data[unit, {data_counter[0]}] which is out of bounds")
                data_matrix[tuple([unit] + indices)] = data[unit, data_counter[0]]
                data_counter[0] += 1
            return

        for i in range(matrix_shape[len(indices)]):
            fill_matrix(data, data_matrix, mask, indices + [i], unit, data_counter)

    for d in range(unit_dim):
        data_counter = [0]  # Using a list to keep it mutable and persistent across recursive calls
        fill_matrix(data, data_matrix, mask, [], d, data_counter)

    if unit_dim == 1:
        data_matrix = np.squeeze(data_matrix, axis=0)

    return data_matrix


def gen_breast_mask(x, y, z, breast_loc=None, R=6, height=10, tkns=0.5, chest_dim=None):
    """
    Generate a breast mask. The breast is modeled as a half ellipsoid.
    This function is adapted from code from Yonghyun and edited & tested by the author.
    :param x: x-axis
    :param y: y-axis
    :param z: z-axis
    :param R: radius of the breast
    :param height: height of the breast
    :param breast_loc: location of the breast tip
    :param tkns: thickness of the coil
    :param chest_dim: dimensions of the chest
    :return: breast mask
    """
    if chest_dim is None:
        chest_dim = [12, 12, 3]
    if breast_loc is None:
        breast_loc = [0, 0, 0]
    loc = breast_loc.copy()
    loc[2] += height
    mask = np.zeros((len(x), len(y), len(z)), dtype=bool)
    # for k in np.arange(np.floor(len(z)/2),len(z), dtype = int):
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                if z[k] < loc[2]:
                    r = np.sqrt(R ** 2 * (1 - ((z[k] - loc[2]) / height) ** 2)) - tkns
                    if (y[j] - loc[1]) ** 2 + (x[i] - loc[0]) ** 2 < r ** 2:
                        mask[i, j, k] = True
                elif z[k] - loc[2] < chest_dim[2]:
                    if abs(x[i] - loc[0]) < chest_dim[0] / 2 and abs(y[j] - loc[1]) < chest_dim[1] / 2:
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


def generate_Y_shape(m, n, thickness=None):
    """
    Generates a matrix of size m x n with a 'Y' shape pattern whose thickness adjusts
    based on the smallest dimension between m and n.

    The 'Y' shape is created with '1's and the rest of the matrix is filled with '0's.
    The top arms of the 'Y' are diagonal lines, and the vertical line starts in the middle.

    Args:
    m (int): Number of rows in the matrix.
    n (int): Number of columns in the matrix.

    Returns:
    np.ndarray: Matrix with the 'Y' shape pattern, with adjustable thickness.

    This function is drafted by ChatGPT on 2024-09-17, edited and tested by the author.
    """

    # Create an m x n matrix filled with zeros
    matrix = np.zeros((m, n), dtype=int)

    # Determine the middle column for the vertical line of the 'Y'
    mid_col = n // 2

    # Calculate the thickness based on the smaller of the dimensions
    if thickness is None:
        thickness = max(1, min(m, n) // 10)  # The thickness is proportional to the size

    # Create the top arms of the 'Y' (diagonal lines)
    for i in range(min(m // 2, n // 2)):
        for t in range(thickness):
            if i + t < m and i + t < n:
                matrix[i + t, i] = 1  # Left diagonal arm (thickened)
                matrix[i + t, n - i - 1] = 1  # Right diagonal arm (thickened)

    # Create the vertical line of the 'Y'
    for i in range(m // 2, m):
        for t in range(-thickness // 2, thickness // 2 + 1):
            if 0 <= mid_col + t < n:
                matrix[i, mid_col + t] = 1

    return matrix
