"""
   Name: algebra.py
   Purpose: Algebraic functions for the project
   Created on: 2/15/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np


def complex2array(x):
    """
    Convert a complex array to a real array.

    Parameters:
    - x (numpy.ndarray): The input complex array.

    Returns:
    - numpy.ndarray: The real array.
    """
    return np.concatenate((np.real(x), np.imag(x)), axis=-1)


def array2complex(x):
    """
    Convert a real array to a complex array.

    Parameters:
    - x (numpy.ndarray): The input real array.

    Returns:
    - numpy.ndarray: The complex array.
    """
    return x[..., :x.shape[-1] // 2] + 1j * x[..., x.shape[-1] // 2:]