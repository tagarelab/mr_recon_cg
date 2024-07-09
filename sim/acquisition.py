"""
   Name: acquisition.py
   Purpose:
   Created on: 2/22/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import algebra as algb

__all__ = ['slice_select', 'B1_effective']


def B1_effective(B1, B0):
    """
    Calculate the effective B1 field
    This function is adapted from a MATLAB function by Github Copilot and edited & tested by the author.

    Parameters:
    - B1 (numpy.ndarray): The B1 field.
    - B0 (numpy.ndarray): The B0 field.

    Returns:
    - numpy.ndarray: The effective B1 field, should have the same shape as B1
    """
    if B1.shape != B0.shape:
        raise ValueError("B1 and B0 must have the same shape.")
    if B1.ndim == 1:
        B1_eff = algb.perpendicular_component(B1, B0)
    else:
        B1_eff = np.zeros(B1.shape)
        for i in range(B1.shape[1]):
            B1_eff[:, i] = algb.perpendicular_component(B1[:, i], B0[:, i])

    return B1_eff


def slice_select(b0_mag, ctr_mag, slc_tkns):
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

    id = (b0_mag > val_1) & (b0_mag < val_2)

    return id
