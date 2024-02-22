"""
   Name: acquisition.py
   Purpose:
   Created on: 2/22/2024
   Created by: Heng Sun
   Additional Notes: 
"""

__all__ = ['slice_select']


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
