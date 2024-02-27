"""
   Name: acquisition.py
   Purpose:
   Created on: 2/22/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np

__all__ = ['slice_select']


def B1_effective(B1, B0):
    """
    Calculate the effective B1 field
    This function is adapted from a MATLAB function by Github Copilot and edited & tested by the author.

    Parameters:
    - B1 (numpy.ndarray): The B1 field.
    - B0 (numpy.ndarray): The B0 field.

    Returns:
    - numpy.ndarray: The effective B1 field.
    """

    return B1 - np.dot(B0, B1) / np.linalg.norm(B0) ** 2 * B0


def slice_select(b0, ctr_mag, slc_tkns):
    """
    Slice selection function
    This function is adapted from a MATLAB function by Github Copilot and edited & tested by the author.

    Parameters:
    - b0 (numpy.ndarray): The B0 field map.
    - ctr_mag (float): The center frequency of the slice.
    - slc_tkns (float): The slice thickness.

    Returns:
    - numpy.ndarray: The slice selection mask.
    """

    val_1 = ctr_mag - slc_tkns / 2
    val_2 = ctr_mag + slc_tkns / 2

    id = (b0 > val_1) & (b0 < val_2)

    return id
