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


def concatenated_to_complex(data, axis=0, mode="real&imag"):
    """
    Convert concatenated real and imaginary data to complex data.
    :param data:
    :param axis:
    :param mode:
    :return:
    """
    data = np.moveaxis(data, axis, -1)
    if mode == "real&imag":
        return data @ np.array([1, 1j])

    elif mode == "mag&phase":
        return data[..., 0] * np.exp(1j * data[..., 1])

    else:
        raise ValueError("Invalid mode. Choose from 'real&imag' or 'mag&phase'.")


def complex_to_concatenated(data, mode="real&imag"):
    """
    Convert complex data to concatenated real and imaginary data.
    :param data:
    :param mode:
    :return:
    """
    if mode == "real&imag":
        return np.array([data.real, data.imag])

    elif mode == "mag&phase":
        return np.array([np.abs(data), np.angle(data)])

    else:
        raise ValueError("Invalid mode. Choose from 'real&imag' or 'mag&phase'.")


def detect_signal(M, B, c, t, T1=None, T2=None):
    """
    Detect the signal at time t
    :param M: the magnetization
    :param B: the net magnetic field: B0 + B_grad that is on during acquisition
    :param c: the sensitivity of the coil
    :param t: the time point
    :param T1: the T1 relaxation time
    :param T2: the T2 relaxation time
    :return:
    """

    # Change coordinates to net B point to z-axis for M and C
    M_uv = algb.rot_mat()  # TODO: take this out of this function so we have the transpose of rotation matrix for the inverse transformation
    # TODO: the final operation should not have M in there

    # Get the perpendicular unit vectors
    u, v = algb.create_perpendicular_unit_vectors(B, M)

    M_u = algb.parallel_component(M, u)
    M_v = algb.parallel_component(M, v)

    # Get the eta and phi values
    eta = get_eta(u, v, c)
    phi = get_phi(u, v, c)

    # Get the d_omega value
    d_omega = np.einsum('ij,ij->j', B, c)

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

    # Ensure input arrays have the same shape
    if u.shape != v.shape or u.shape != c.shape:
        raise ValueError("Input vectors u, v, and c must have the same dimensions")

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

    # Ensure input arrays have the same shape
    if u.shape != v.shape or u.shape != c.shape:
        raise ValueError("Input vectors u, v, and c must have the same dimensions")

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
    :param t: 1D array with length n
    :param eta: 1D array with length m
    :param d_omega: 1D array with length m
    :param phi: 1D array with length m
    :param T1: Optional, not used in current function
    :param T2: 1D array with length m
    :return: 2D array with shape (m, n)
    """

    # Ensure input arrays are 1D
    t = np.atleast_1d(t)
    eta = np.atleast_1d(eta)
    d_omega = np.atleast_1d(d_omega)
    phi = np.atleast_1d(phi)

    if T2 is not None:
        T2 = np.atleast_1d(T2)

    # Validate input dimensions
    if not (len(eta) == len(d_omega) == len(phi) and (T2 is None or len(T2) == len(eta))):
        raise ValueError("eta, d_omega, phi, and T2 must be vectors of the same length")

    # Use broadcasting to calculate the signal matrix
    signal = 0.5 * eta[:, np.newaxis] * np.exp(
        1j * (d_omega[:, np.newaxis] * t[np.newaxis, :] - phi[:, np.newaxis]))

    if T2 is not None:
        signal *= np.exp(-t[np.newaxis, :] / T2[:, np.newaxis])

    # sum up the signal from all voxels
    signal = np.sum(signal, axis=0)

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
