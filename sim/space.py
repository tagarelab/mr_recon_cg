"""
   Name: space.py
   Purpose:
   Created on: 10/7/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import algebra as algb

__all__ = []


def swapaxes(data, axis1, axis2):
    """
    Swap two axes of a 3D data array (size 3*x*x*x).
    :param data: 3D data array
    :param axis1: axis to swap with axis2
    :param axis2: axis to swap with axis1

    :return: 3D data array with axis1 and axis2 swapped
    """

    # Check if axis1 and axis2 are valid
    # assert axis1 in [0, 1, 2], "Axis1 must be 0, 1, or 2"
    # assert axis2 in [0, 1, 2], "Axis2 must be 0, 1, or 2"
    assert axis1 != axis2, "Axis1 and Axis2 must be different"

    # Swap the axes
    if data.shape[0] == 3 and len(data.shape) == 4:  # if it has 3D information
        temp = data.copy()
        data[axis1 - 1, :, :, :] = temp[axis2 - 1, :, :, :]
        data[axis2 - 1, :, :, :] = temp[axis1 - 1, :, :, :]
    swapped_data = np.swapaxes(data, axis1, axis2)

    return swapped_data
