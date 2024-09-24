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


def detect_signal(M, B, c, t, T1=None, T2=None):
    """
    Detect the signal at time t
    :param M: the magnetization
    :param B: the magnetic field
    :param c: the sensitivity of the coil
    :param t: the time point
    :param T1: the T1 relaxation time
    :param T2: the T2 relaxation time
    :return:
    """

    # Get the perpendicular unit vectors
    u, v = algb.create_perpendicular_unit_vectors(B)

    # Get the eta and phi values
    eta = get_eta(u, v, c)
    phi = get_phi(u, v, c)

    # Get the d_omega value
    d_omega = np.dot(B, c)

    # Get the signal
    signal = detection(t, eta, d_omega, phi, T1, T2)

    return signal


def get_eta(u, v, c):
    """
    Calculate the eta value.
    :param u: one perpendicular unit vector in the plane perpendicular to the B field.
              Can be a 1D array of size 3 or a 2D array of size (3, n).
    :param v: the other perpendicular unit vector in the plane perpendicular to the B field.
              Can be a 1D array of size 3 or a 2D array of size (3, n).
    :param c: the sensitivity of the coil.
              Can be a 1D array of size 3 or a 2D array of size (3, n).
    :return: eta values.
    """

    # Ensure arrays are 2D for consistent processing
    if u.ndim == 1:
        u = u[:, np.newaxis]
    if v.ndim == 1:
        v = v[:, np.newaxis]
    if c.ndim == 1:
        c = c[:, np.newaxis]

    # Calculate the dot products
    dot_u_c = np.einsum('ij,ij->j', u, c)
    dot_v_c = np.einsum('ij,ij->j', v, c)

    # Calculate eta
    eta = np.sqrt(dot_u_c ** 2 * dot_v_c ** 2)

    return eta


def get_phi(u, v, c):
    """
    Calculate the phi value.
    :param u: one perpendicular unit vector in the plane perpendicular to the B field.
              Can be a 1D array of size 3 or a 2D array of size (3, n).
    :param v: the other perpendicular unit vector in the plane perpendicular to the B field.
              Can be a 1D array of size 3 or a 2D array of size (3, n).
    :param c: the sensitivity of the coil.
              Can be a 1D array of size 3 or a 2D array of size (3, n).
    :return: phi values.
    """

    # Ensure arrays are 2D for consistent processing
    if u.ndim == 1:
        u = u[:, np.newaxis]
    if v.ndim == 1:
        v = v[:, np.newaxis]
    if c.ndim == 1:
        c = c[:, np.newaxis]

    # Calculate the dot products
    dot_u_c = np.einsum('ij,ij->j', u, c)
    dot_v_c = np.einsum('ij,ij->j', v, c)

    # Calculate phi
    phi = np.arctan2(dot_u_c, dot_v_c)

    return phi


def detection(t, eta, d_omega, phi, T1=None, T2=None):
    """
    Calculate the signal at time t
    :param t:
    :param eta:
    :param d_omega:
    :param phi:
    :param T1: #TODO: add function for T1 dephasing?
    :param T2:
    :return:
    """
    signal = 0.5 * eta * np.exp(1j * (d_omega * t - phi))
    if T2 is not None:
        signal = signal * np.exp(-t / T2)
    return signal


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
