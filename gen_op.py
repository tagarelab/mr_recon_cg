"""
   Name: gen_op.py
   Purpose:
   Created on: 2/22/2024
   Created by: Heng Sun
   Additional Notes:
"""

from optlib import operators as op
import algebra as algb


def spin_density_operator(n, mask=None):
    """
    Create a spin density operator.
    This function is drafted by Github Copilot and edited & tested by the author.

    Parameters:
    - n (int): The number of spins.

    Returns:
    - numpy.ndarray: The spin density operator.
    """

    return op.hadamard_op(n)


def polarization_operator(B0_selected):
    """
    Create a polarization operator.
    This function is drafted by Github Copilot and edited & tested by the author.

    Parameters:
    - n (int): The number of rows in the operator.
    - m (int): The number of columns in the operator.

    Returns:
    - numpy.ndarray: The polarization operator.
    """

    return op.hadamard_op(B0_selected)
