"""
   Name: comb_test_240325.py
   Purpose:
   Created on:
   Created by: Heng Sun
   Additional Notes: 
"""

# Comb EMI
# %%
## Packages and Data
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from optlib import c_grad as cg
from optlib import operators as op


# %%
## Define supporting functions
def mask_mat(mask_arr):
    num_true = sum(mask_arr)
    num_all = len(mask_arr)
    mask_matrix = np.zeros((num_true, num_all), dtype=bool)
    j = 0
    for i in range(num_all):
        if mask_arr[i] == True:
            mask_matrix[j, i] = True
            j += 1
    return mask_matrix


# This function is drafted by ChatGPT on 2023-12-16, edited and tested by the author.
def im2freq(image):
    """
    Convert the input image to the frequency domain.

    Parameters:
    - image (numpy.ndarray): The input image.

    Returns:
    - numpy.ndarray: The image in the frequency domain.
    """
    # F = sgp.linop.FFT(image.shape)
    # return F(image)  # k-space
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))  # cartesian


# This function is drafted by ChatGPT on 2023-12-16, edited and tested by the author.
def freq2im(frequency_data):
    """
    Convert the input frequency domain data to an image.

    Parameters:
    - frequency_data (numpy.ndarray): The input data in the frequency domain.

    Returns:
    - numpy.ndarray: The reconstructed image.
    """
    # IF = sgp.linop.IFFT(frequency_data.shape)
    # return IF(frequency_data)  # k-space
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(frequency_data)))  # cartesian


# This function is drafted by ChatGPT on 2023-12-16, edited and tested by the author.
def rmse(y_true, y_pred, axis=None):
    """
    Calculate the Root Mean Square Error (RMSE) between two arrays.

    Parameters:
    - y_true (numpy.ndarray): The true values.
    - y_pred (numpy.ndarray): The predicted values.
    - axis (int, optional): Axis or axes along which the RMSE is calculated. Default is None.

    Returns:
    - numpy.ndarray: The RMSE values.
    """
    mse = np.mean((y_true - y_pred) ** 2, axis=axis)
    rmse_values = np.sqrt(mse)
    return rmse_values


def gen_sn(sn, length, dt):
    """
    Generate structured noise.

    Parameters:
    - sn (numpy.ndarray): Structured noise parameters (amplitude, frequency, phase).
    - length (int): Length of the noise.

    Returns:
    - numpy.ndarray: Generated structured noise.
    """
    tt = np.arange(0, length) * dt
    noi_gen = np.zeros_like(tt, dtype=complex)
    for i in range(sn.shape[1]):
        noi_gen += sn[0, i] * np.exp(1j * (2 * np.pi * sn[1, i] * tt + sn[2, i]))

    return noi_gen


# def gen_sn(signal, N_echoes, TE, dt, sn, polar_time=0):
#     """
#     Generate complex structured noise and inject it into the signal.
#
#     Parameters:
#     - signal (numpy.ndarray): Input signal.
#     - N_echoes (int): Number of echoes.
#     - TE (float): Echo time.
#     - dt (float): Sampling interval.
#     - sn (numpy.ndarray): Structured noise parameters (amplitude, frequency, phase).
#
#     Returns:
#     - numpy.ndarray: Generated structured noise (same t span like the input signal)
#     - numpy.ndarray: Generated structured noise (continuous)
#     """
#
#     if polar_time > 0:
#         signal_te = remove_polarization(signal, polar_time, dt)
#     else:
#         signal_te = signal
#     # Generate the full signal
#     zfilled_data = np.reshape(signal_te, (-1, N_echoes))
#     acq_len = zfilled_data.shape[0]
#
#     # Generate complex structured noise
#     tt = np.arange(0, N_echoes * TE, dt)
#     noi_gen = np.zeros_like(tt, dtype=complex)
#
#     for i in range(sn.shape[1]):
#         noi_gen += sn[0, i] * np.exp(1j * (2 * np.pi * sn[1, i] * tt + sn[2, i]))
#
#     # Visualize
#     # plt.figure()
#     # plt.plot(tt, np.real(noi_gen), label='Real')
#     # plt.plot(tt, np.imag(noi_gen), label='Imaginary')
#     # plt.title('Injected EMI')
#     # plt.legend()
#     # plt.xlabel('Time (s)')
#     # plt.ylabel('Amplitude')
#     # plt.show()
#
#     # Sample it the same way as the signal
#     if polar_time > 0:
#         noi_te = noi_gen[int(polar_time / dt):]
#         noisy_data_mat = np.reshape(noi_te, (-1, N_echoes))
#         noisy_data = noisy_data_mat[:acq_len, :].reshape(1, -1)
#         noisy_data = np.concatenate(noi_gen[:, int(polar_time / dt)], noisy_data)
#     else:
#         noisy_data_mat = np.reshape(noi_gen, (-1, N_echoes))
#         noisy_data = noisy_data_mat[:acq_len, :].reshape(1, -1)
#
#     return noisy_data, noi_gen


class ifft_op:
    """ Inverse fft operator. Takes a complex
        argument as input and produces a real
        output.
    """

    def __init__(self):
        pass

    def forward(self, x):
        return np.fft.ifftn(np.fft.ifftshift(x))

    #        return np.real(np.fft.ifftn(np.fft.ifftshift(x)))
    def transpose(self, x):
        #        return np.fft.fftshift(np.fft.fftn(np.real(x)))
        return np.fft.fftshift(np.fft.fftn(x))


class selection_op:
    """ Selection operator. Given a vector as an input it
        produces as an output certain components of the vector.
        See the __init__ and forward methods for details
    """

    def __init__(self, x_shape, idx):
        """ x_shape is the full shape of the input array for forward
            idx is the set of indices from which the components of
                x are extracted to make the output
            idx is required to be produced by np.r_ (see the use below)
            For now x is assumed to be 1-dimensional
        """
        self.x_shape = x_shape
        self.idx = idx

    def forward(self, x):
        return x[self.idx]

    def transpose(self, x):
        z = np.zeros(self.x_shape, dtype=np.dtype(x[0]))
        z[self.idx] = x
        return z


def s_thresh(a, alpha):
    """ Soft thresholding needed for ADMM
    """
    return (np.maximum(abs(a), alpha) - alpha) * a / np.abs(a)


def h_thresh(a, alpha):
    """ Hard thresholding needed for ADMM
    """
    return (abs(a) >= alpha) * a


def norm(x):
    """ Returns the norm of x. Works for complex x
        Eventually to be replaced with the norm in
            the cg module
    """
    return np.sqrt(np.sum(np.abs(x) ** 2))


def admm_l2_l1(A, b, x0, l1_wt=1.0, rho=1.0, iter_max=100, eps=1e-2):
    """ADMM L2 L1 function. Minimizes \|Ax-b\|^2 + \lambda \|x\|_1
       Input:
       A: operator, b as described in the formula above. x0 is the initial value of x
       l1_wt: This is lambda (I am not using lambda, because that is a protected
                                   word in Python)
       rho: Augmented Lagrangian parameter for ADMM
       iter_max: maximum number of iterations
       eps: stopping criterion (needs to be more sophisticated)

       Returns the minimizing x, and a flag indicated whether iter_max was reached
    """
    # Initialize variables
    xk = x0
    zk = np.zeros_like(x0)
    uk = np.zeros_like(x0)

    # initialize iteration
    B = op.scalar_prod_op(rho)
    mu = 10
    rk_1 = np.inf
    sk_1 = np.inf
    steps = 0

    # Iterate till termination
    while ((steps < iter_max) & ((norm(rk_1) > eps) | (norm(sk_1) > eps))):
        xk_1, cg_flag = cg.solve_lin_cg(b, A, xk, B=B, c=rho * (zk - uk), max_iter=20, inner_max_iter=6)
        zk_1 = s_thresh(xk_1 + uk, l1_wt / rho)
        uk_1 = uk + xk_1 - zk_1
        rk_1 = xk_1 - zk_1
        sk_1 = rho * (zk - zk_1)

        # Update variables
        xk = xk_1
        zk = zk_1
        uk = uk_1
        steps = steps + 1

        # Update rho
        # You may comment this out if you are adjusting rho manually
        if norm(rk_1) > mu * norm(sk_1):
            rho = rho * 1.5
        elif norm(sk_1) > mu * norm(rk_1):
            rho = rho / 1.5

    return xk, steps < iter_max


# %% Define comb_optimized: the noise cancellation function
def comb_optimized(signal, N_echoes, TE, dt, lambda_val, step, tol, max_iter, pre_drop, post_drop, pk_win,
                   polar_time=0):
    """
    Comb noise self-correction method

    Parameters:
    - signal (numpy.ndarray): Input signal.
    - N_echoes (int): Number of echoes.
    - TE (float): Echo time.
    - dt (float): Time step.
    - lambda_val (float): Regularization term.
    - step (float): Step size.
    - iter (int): Number of iterations.
    - pre_drop (int): Drop this many points at the front of each echo.
    - post_drop (int): Drop this many points at the end of each echo.
    - pk_win (float): Window size for peak recognition.

    Returns:
    - acq_corr (numpy.ndarray): Corrected signal.
    """
    # Parameter formatting
    signal = signal.T
    pre_drop = np.uint16(pre_drop)
    post_drop = np.uint16(post_drop)

    # Set binary mask for sampling window
    TE_len = np.uint16(TE / dt)
    rep_len = signal.shape[0]
    acq_len = np.uint16(rep_len / N_echoes)

    samp_1Echo = np.concatenate([np.ones(acq_len, dtype=bool), np.zeros(TE_len - acq_len, dtype=bool)])
    samp_all = np.tile(samp_1Echo, N_echoes)

    # Auto peak recognition to create comb
    sig_len_hf = np.uint16((acq_len - pre_drop - post_drop) * pk_win / 2)

    # Pick out the peak location
    pks = np.abs(np.reshape(signal, (-1, N_echoes)))
    pks[:16, :] = 0  # some time there's a leak of signal at the beginning
    pks_val, pks_id = np.max(pks, axis=0), np.argmax(pks, axis=0)
    max_pks_id = np.argpartition(pks_val, -10)[-10:]
    pk_id = np.uint16(np.mean(pks_id[max_pks_id]))

    pk_id = 50  # temp solution for debug

    # Signal (echo peak) part only
    sig_1Echo = np.zeros(acq_len, dtype=bool)
    sig_1Echo[max(0, pk_id - sig_len_hf):min(pk_id + sig_len_hf, acq_len)] = 1
    sig_all = np.tile(np.concatenate([sig_1Echo, np.zeros(TE_len - acq_len, dtype=bool)]), N_echoes)

    # Noise part only
    noi_1Echo = np.ones(acq_len, dtype=bool)
    noi_1Echo[max(0, pk_id - sig_len_hf):min(pk_id + sig_len_hf, acq_len)] = 0
    # noi_1Echo[:pre_drop] = 0
    # noi_1Echo[-post_drop:] = 0
    noi_all = np.tile(np.concatenate([noi_1Echo, np.zeros(TE_len - acq_len, dtype=bool)]), N_echoes)

    # Simulate a signal
    Nc = 1  # Number of coils

    # Time domain
    t = np.arange(len(samp_all)) - len(samp_all) / 2

    # Set dimensions
    x_dim, y_dim = len(t), 3
    k1, k2 = np.arange(-x_dim / 2, x_dim / 2), np.arange(-y_dim / 2, y_dim / 2)

    # Get undersampled data and k-space
    samp_data = signal.flatten('F')
    zfilled_data = np.reshape(samp_data, (acq_len, -1))
    zfilled_data = np.concatenate([zfilled_data, np.zeros((TE_len - acq_len, N_echoes))], axis=0)
    zfilled_data = zfilled_data.flatten('F')
    noi_data = zfilled_data[noi_all]
    sig_data = zfilled_data[sig_all]
    samp_data = zfilled_data[samp_all]

    # Choose the mask to input
    input_mask = noi_all
    data_us = zfilled_data[input_mask]
    plt.figure()
    plt.plot(np.abs(noi_all), label="Noise Mask")
    plt.plot(np.abs(sig_all), label="Signal Mask")
    plt.plot(np.abs(samp_all), label="Sample Mask")
    plt.title("Masks")
    plt.legend()
    plt.show()

    # Predict the EMI

    emi_prdct = sn_recognition(signal=zfilled_data, mask=input_mask, lambda_val=lambda_val, stepsize=step, tol=tol,
                               max_iter=max_iter,
                               method="conj_grad_l1_reg")
    # emi_prdct = np.squeeze(emi_prdct)
    # factor = np.linalg.norm(noi_data, 1) / np.linalg.norm(emi_prdct[noi_all], 1)
    # emi_prdct *= factor
    # emi_prdct = np.abs(emi_prdct) * np.exp(-1j * np.angle(emi_prdct))

    # Get the output
    acq_corr = samp_data - emi_prdct[samp_all]

    plt.figure()
    plt.plot(np.abs(samp_data), label="Before correction")
    plt.plot(np.abs(acq_corr), label="After correction")
    plt.title("After correction")
    plt.legend()
    plt.show()

    return emi_prdct[samp_all]


def sampled_to_full(signal, polar_time, dt, acq_len, N_echoes, TE_len):
    polar_period = calculate_polar_period(polar_time=polar_time, dt=dt)
    signal = signal.flatten('F')
    signal_pol = signal[:polar_period]
    signal_te = signal[polar_period:]

    # Get undersampled data and k-space
    zfilled_data = np.reshape(signal_te, (acq_len, -1))
    zfilled_data = np.concatenate([zfilled_data, np.zeros((TE_len - acq_len, N_echoes))], axis=0)
    zfilled_data = zfilled_data.flatten('F')

    # add polarization time
    zfilled_data = np.concatenate([signal_pol, zfilled_data], axis=0)
    return zfilled_data


def gen_sig_mask(acq_len, N_echoes, TE_len, pk_id, sig_len_hf, polar_time, dt):
    sig_1Echo = np.zeros(acq_len, dtype=bool)
    sig_1Echo[max(0, pk_id - sig_len_hf):min(pk_id + sig_len_hf, acq_len)] = 1
    sig_all = np.tile(np.concatenate([sig_1Echo, np.zeros(TE_len - acq_len, dtype=bool)]), N_echoes)
    sig_all = add_polarization(sig_all, polar_time, dt, value=0, type=bool)
    return sig_all.astype(bool)


def gen_noi_mask(acq_len, N_echoes, TE_len, pk_id, sig_len_hf, polar_time, dt):
    noi_1Echo = np.ones(acq_len, dtype=bool)
    noi_1Echo[max(0, pk_id - sig_len_hf):min(pk_id + sig_len_hf, acq_len)] = 0
    noi_all = np.tile(np.concatenate([noi_1Echo, np.zeros(TE_len - acq_len, dtype=bool)]), N_echoes)
    noi_all = add_polarization(noi_all, polar_time, dt, value=1, type=bool)
    return noi_all.astype(bool)


def gen_samp_mask(acq_len, N_echoes, TE_len, polar_time, dt, pol=False):
    samp_1Echo = np.concatenate([np.ones(acq_len, dtype=bool), np.zeros(TE_len - acq_len, dtype=bool)])
    samp_all = np.tile(samp_1Echo, N_echoes)
    samp_all = add_polarization(signal=samp_all, time=polar_time, dt=dt, value=pol, type=bool)
    return samp_all.astype(bool)


def add_polarization(signal, time, dt, value=0, type=complex):
    """
    Add polarization time to the signal

    Parameters:
    - signal (numpy.ndarray): Input signal.
    - time (numpy.ndarray): Time vector.
    - dt (float): Time step.

    Returns:
    - numpy.ndarray: Polarized signal.
    """
    if signal.ndim == 2:
        polar = np.ones((1, calculate_polar_period(polar_time=time, dt=dt)), dtype=type) * value
        signal = np.concatenate((polar, signal), axis=1)
    elif signal.ndim == 1:
        polar = np.ones(calculate_polar_period(polar_time=time, dt=dt), dtype=type) * value
        signal = np.concatenate((polar, signal))

    return signal


def remove_polarization(signal, time, dt):
    """
    Remove polarization time from the signal

    Parameters:
    - signal (numpy.ndarray): Input signal.
    - time (numpy.ndarray): Time vector.
    - dt (float): Time step.

    Returns:
    - numpy.ndarray: Polarized signal.
    """
    return signal[calculate_polar_period(polar_time=time, dt=dt):]


def calculate_polar_period(polar_time, dt):
    return int(np.ceil(polar_time / dt))


# %% md
## Define main optimization function
# %%
# class cg_comb(sgp.app.App):
#     def __init__(self, lambda_val, mask, y, max_iter, stepsize, tol, **kwargs):
#         self.x = np.zeros((mask.shape[0], 1), np.complex128)
#         self.y = np.expand_dims(y[mask], axis=1)
#         # mask_matrix = np.fft.fft(np.eye(len(y)))[mask, :]
#         mask_matrix = mask_mat(mask)
#
#         self.F = sgp.linop.FFT(self.x.shape, center=True, axes=(-1, -2))
#         M = sgp.linop.MatMul(ishape=self.x.shape, mat=mask_matrix)
#         self.A = M * self.F.H
#
#         proxg = sgp.prox.L1Reg(self.A.ishape, lambda_val)
#
#         def gradf(x):
#             return self.A.H * (self.A * x - self.y)
#
#         alg = sgp.alg.GradientMethod(gradf=gradf, x=self.x, alpha=stepsize, proxg=proxg, accelerate=False,
#                                      max_iter=max_iter,
#                                      tol=tol, **kwargs)
#         # alg = sgp.alg.ConjugateGradient(A=self.A, b=self.y, x=self.x, P=proxg, max_iter=max_iter, tol=0)
#         super().__init__(alg)
#
#     def _output(self):
#         # return self.F.H(self.x)
#         return self.x


# %%
def sn_recognition(signal, mask, lambda_val, tol=0.1, stepsize=1, max_iter=100, method="conj_grad_l1_reg"):
    mask_matrix = np.fft.fft(np.eye(len(signal)))[mask, :]
    y = np.multiply(mask, signal)
    if method == "conj_grad_l1_reg":
        print("Conjugate Gradient method with L1 regularization")
        mask_id = np.where(mask)
        S = selection_op(signal.shape, mask_id)
        F = ifft_op()
        A = op.composite_op(S, F)
        sn_prdct, cg_flag = admm_l2_l1(A=A, b=signal[mask], x0=np.zeros_like(y), l1_wt=lambda_val, rho=1.0,
                                       iter_max=max_iter,
                                       eps=tol)
        # sn_prdct = cg_comb(lambda_val=lambda_val, mask=mask, y=y, max_iter=max_iter, stepsize=stepsize, tol=tol).run()
        if not cg_flag:
            print("Step limit reached.")

        # plt.figure()
        # plt.plot(np.abs(sn_prdct))
        # plt.title("EMI prediction in frequency domain")
        # plt.show()
        #
        sn_prdct = F.forward(sn_prdct)
        # plt.figure()
        # plt.plot(np.abs(sn_prdct))
        # plt.title("EMI prediction in time domain")
        # plt.show()
    else:
        print("Invalid optimization method.")

    return sn_prdct
