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
from skimage.transform import radon, iradon

import visualization
from optlib import c_grad as cg
from optlib import operators as op


# %%
## Define supporting functions
def freq_axis(N, dt):
    freq_axis = sp.fft.fftshift(sp.fft.fftfreq(N, dt))
    return freq_axis


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
def im2freq(image, theta=np.arange(180)):
    """
    Convert the input image to the frequency domain.

    Parameters:
    - image (numpy.ndarray): The input image.

    Returns:
    - numpy.ndarray: The image in the frequency domain.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))  # cartesian
    # return radon_fft(image, theta)  # radon


# This function is drafted by ChatGPT on 2023-12-16, edited and tested by the author.
def freq2im(frequency_data, theta=np.arange(180)):
    """
    Convert the input frequency domain data to an image.

    Parameters:
    - frequency_data (numpy.ndarray): The input data in the frequency domain.

    Returns:
    - numpy.ndarray: The reconstructed image.
    """
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(frequency_data)))  # cartesian
    # return ifft_iradon(frequency_data, theta=theta) # radon


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


def percent_residue(pred_signal, true_signal):
    """
    Calculate the percentage of the signal that was cancelled.

    Parameters:
    - pred_signal (numpy.ndarray): The predicted signal.
    - true_signal (numpy.ndarray): The true signal.

    Returns:
    - float: The percentage of the signal that was cancelled.
    """
    return 100 * np.linalg.norm(true_signal - pred_signal, 2) / np.linalg.norm(true_signal, 2)


def rand_sn(N_sn, amp_range, freq_range, phase_range):
    """
    Generate random structured noise parameters.

    Parameters:
    - N_sn (int): Number of structured noise parameters.
    - amp_range (tuple): Range of the amplitude.
    - freq_range (tuple): Range of the frequency.
    - phase_range (tuple): Range of the phase.

    Returns:
    - numpy.ndarray: Random structured noise parameters.
    """
    sn = np.zeros((3, N_sn))
    sn[0, :] = np.random.uniform(amp_range[0], amp_range[1], N_sn)
    sn[1, :] = np.random.uniform(freq_range[0], freq_range[1], N_sn)
    sn[2, :] = np.random.uniform(phase_range[0], phase_range[1], N_sn)
    return sn


def rand_sn_from_list(N_sn, amp_list, freq_list, phase_list):
    """
    Generate random structured noise parameters.

    Parameters:
    - N_sn (int): Number of structured noise parameters.
    - amp_range (tuple): Range of the amplitude.
    - freq_range (tuple): Range of the frequency.
    - phase_range (tuple): Range of the phase.

    Returns:
    - numpy.ndarray: Random structured noise parameters.
    """
    sn = np.zeros((3, N_sn))
    sn[0, :] = np.random.choice(amp_list, N_sn)
    sn[1, :] = np.random.choice(freq_list, N_sn)
    sn[2, :] = np.random.choice(phase_list, N_sn)
    return sn


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


def gen_white_noise(wgn_lin, shape):
    return np.random.normal(0, wgn_lin, shape) + 1j * np.random.normal(0, wgn_lin, shape)

def unpad(array, pad_width):
    return array[pad_width:-pad_width]


def ifft_iradon(phantom_fft, theta):
    """
    First ifft then iradon transformation
    First dim of phantom_fft is the signal dim, second is angle span dim
    """
    phantom_rad = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(phantom_fft)))
    phantom_img = iradon(np.abs(phantom_rad), theta)
    return phantom_img


def radon_fft(img, theta):
    """
    First radon then fft transformation
    First dim of phantom_fft is the signal dim, second is angle span dim
    """
    phantom_rad = radon(img, theta, circle=True, preserve_range=True)
    phantom_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(phantom_rad)))
    return phantom_fft


class ifft_op:
    """ Inverse fft operator. Takes a complex
        argument as input and produces a real
        output.
    """

    def __init__(self):
        self.pad_width = 1

    def forward(self, x):
        return np.fft.ifftn(np.fft.ifftshift(x))

        # padded
        # return unpad(np.fft.ifftn(np.fft.ifftshift(np.pad(x, pad_width=self.pad_width, mode='constant'))),
        #              pad_width=self.pad_width)

        # return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x)))

    def transpose(self, x):
        return np.fft.fftshift(np.fft.fftn(x))

        # padded
        # return unpad(np.fft.fftshift(np.fft.fftn(np.pad(x, pad_width=self.pad_width, mode='constant'))),
        #              pad_width=self.pad_width)

        # return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x)))


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
    return (np.maximum(np.abs(a), alpha) - alpha) * a / np.abs(a)


def h_thresh(a, alpha):
    """ Hard thresholding needed for ADMM
    """
    return (np.abs(a) >= alpha) * a


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
        # zk_1 = h_thresh(xk_1 + uk, l1_wt / rho)
        uk_1 = uk + xk_1 - zk_1
        rk_1 = xk_1 - zk_1
        sk_1 = rho * (zk - zk_1)

        # visualization.complex(xk_1, name="xk_1")
        # visualization.complex(zk_1, name="zk_1")
        # visualization.complex(uk_1, name="uk_1")
        # visualization.complex(rk_1, name="rk_1")
        # visualization.complex(sk_1, name="sk_1")

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
                   polar_time=0, rho=1.0):
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
    polar_period = calculate_polar_period(polar_time=polar_time, dt=dt)
    # signal_pol = signal[:, :polar_period]
    signal_te = signal[polar_period:].T
    pre_drop = np.uint16(pre_drop)
    post_drop = np.uint16(post_drop)

    # Set binary mask for sampling window
    TE_len = np.uint16(TE / dt)
    rep_len = signal_te.shape[0]
    acq_len = np.uint16(rep_len / N_echoes)

    # samp_1Echo = np.concatenate([np.ones(acq_len, dtype=bool), np.zeros(TE_len - acq_len, dtype=bool)])
    # samp_all = np.tile(samp_1Echo, N_echoes)

    # Auto peak recognition to create comb
    sig_len_hf = np.uint16((acq_len - pre_drop - post_drop) * pk_win / 2)

    # Pick out the peak location
    pks = np.abs(np.reshape(signal_te, (N_echoes, -1)))
    pks[:16, :] = 0  # some time there's a leak of signal at the beginning
    pks_val, pks_id = np.max(pks, axis=0), np.argmax(pks, axis=0)
    max_pks_id = np.argpartition(pks_val, -10)[-10:]
    pk_id = np.uint16(np.mean(pks_id[max_pks_id]))

    pk_id = sig_len_hf  # temp solution for debug

    # Generate masks
    noi_all = gen_noi_mask(acq_len, N_echoes, TE_len, pk_id, sig_len_hf, polar_time, dt)
    sig_all = gen_sig_mask(acq_len, N_echoes, TE_len, pk_id, sig_len_hf, polar_time, dt)
    samp_all = gen_samp_mask(acq_len, N_echoes, TE_len, polar_time, dt)

    # Signal (echo peak) part only
    # sig_1Echo = np.zeros(acq_len, dtype=bool)
    # sig_1Echo[max(0, pk_id - sig_len_hf):min(pk_id + sig_len_hf, acq_len)] = 1
    # sig_all = np.tile(np.concatenate([sig_1Echo, np.zeros(TE_len - acq_len, dtype=bool)]), N_echoes)

    # Noise part only
    # noi_1Echo = np.ones(acq_len, dtype=bool)
    # noi_1Echo[max(0, pk_id - sig_len_hf):min(pk_id + sig_len_hf, acq_len)] = 0
    # # noi_1Echo[:pre_drop] = 0
    # # noi_1Echo[-post_drop:] = 0
    # noi_all = np.tile(np.concatenate([noi_1Echo, np.zeros(TE_len - acq_len, dtype=bool)]), N_echoes)

    # Simulate a signal
    # Nc = 1  # Number of coils

    # Time domain
    # t = np.arange(len(samp_all)) - len(samp_all) / 2

    # Set dimensions
    # x_dim, y_dim = len(t), 3
    # k1, k2 = np.arange(-x_dim / 2, x_dim / 2), np.arange(-y_dim / 2, y_dim / 2)

    # # Get undersampled data and k-space
    # samp_data = signal_te.flatten('F')
    # zfilled_data = np.reshape(samp_data, (acq_len, -1))
    # zfilled_data = np.concatenate([zfilled_data, np.zeros((TE_len - acq_len, N_echoes))], axis=0)
    # zfilled_data = zfilled_data.flatten('F')
    #
    # # add polarization time
    # zfilled_data = np.concatenate([signal_pol, zfilled_data], axis=1)

    zfilled_data = sampled_to_full(signal, polar_time, dt, acq_len, N_echoes, TE_len)
    # visualization.complex(zfilled_data, name="zfilled_data")

    # masked data
    noi_data = zfilled_data[noi_all]
    # sig_data = zfilled_data[sig_all]
    samp_data = zfilled_data[samp_all]

    # Visualize the masks
    # plt.figure()
    # plt.plot(np.abs(noi_all), label="Noise Mask")
    # plt.plot(np.abs(sig_all), label="Signal Mask")
    # plt.plot(np.abs(samp_all), label="Sample Mask")
    # plt.title("Masks")
    # plt.legend()
    # plt.show()

    pol_mask = gen_pol_mask(N_echoes, TE_len, polar_time, dt)
    all_ones_mask = np.ones_like(samp_all)

    input_mask = noi_all
    # plt.figure()
    # plt.plot(np.abs(input_mask))
    # plt.title("Input Mask")
    # plt.show()

    # Predict the EMI
    emi_prdct_tot = np.zeros_like(zfilled_data)

    # Multi-loop Auto lambda
    if lambda_val == -1:
        lambda_val = auto_lambda(zfilled_data, rho)
    while lambda_val > 0:
        emi_prdct = sn_recognition(signal=zfilled_data, mask=input_mask, lambda_val=lambda_val, stepsize=step, tol=tol,
                                   max_iter=max_iter,
                                   method="conj_grad_l1_reg", rho=rho)
        emi_prdct_tot += emi_prdct
        zfilled_data[samp_all] = zfilled_data[samp_all] - emi_prdct[samp_all]
        lambda_val = auto_lambda(zfilled_data, rho, lambda_default=lambda_val)

    # Single-loop Auto lambda
    # lambda_val = auto_lambda(zfilled_data, rho, lambda_default=99999999)
    # emi_prdct_tot = sn_recognition(signal=zfilled_data, mask=input_mask, lambda_val=lambda_val, stepsize=step, tol=tol,
    #                                max_iter=max_iter,
    #                                method="conj_grad_l1_reg", rho=rho)

    # emi_prdct = np.squeeze(emi_prdct)
    # factor = np.linalg.norm(noi_data, 1) / np.linalg.norm(emi_prdct[noi_all], 1)
    # emi_prdct *= factor
    # emi_prdct = np.abs(emi_prdct) * np.exp(-1j * np.angle(emi_prdct))

    # Get the output
    # acq_corr = samp_data - emi_prdct[samp_all]
    #
    # plt.figure()
    # plt.plot(np.abs(samp_data), label="Before correction")
    # plt.plot(np.abs(acq_corr), label="After correction")
    # plt.title("After correction")
    # plt.legend()
    # plt.show()

    result = emi_prdct_tot
    # result = add_polarization(emi_prdct, polar_time, dt)

    return result


def auto_lambda(signal, rho, lambda_default=np.inf, tol=0.4, cvg=0.95, ft_prtct=20):
    """
    Automatically determine the lambda value for the signal.

    Parameters:
    - signal (numpy.ndarray): Input signal.
    - rho (float): Augmented Lagrangian parameter for ADMM.

    Returns:
    - float: The lambda value.
    - bool: No need for another iteration
    """
    lambda_val = np.max(np.abs(np.fft.fft(signal))) / rho * tol
    if lambda_val > np.mean(np.abs(np.fft.fft(signal))) / rho * ft_prtct and lambda_val < lambda_default * cvg:
        # print("avg sig: ", np.mean(np.abs(np.fft.fft(signal))) / rho)
        # print("Auto lambda: ", lambda_val)
        return lambda_val
    return -1



def sampled_to_full(signal, polar_time, dt, acq_len, N_echoes, TE_len):
    polar_period = calculate_polar_period(polar_time=polar_time, dt=dt)
    signal = signal.flatten('F')  # TODO: risk of error
    signal_pol = signal[:polar_period]
    signal_te = signal[polar_period:]

    # Get undersampled data and k-space
    zfilled_data = np.reshape(signal_te, (N_echoes, -1))
    # visualization.imshow(np.real(zfilled_data), name="zfilled_data")
    zfilled_data = np.concatenate([zfilled_data, np.zeros((N_echoes, TE_len - acq_len))], axis=1)
    # visualization.imshow(np.real(zfilled_data), name="zfilled_data")
    zfilled_data = zfilled_data.flatten()
    # visualization.complex(zfilled_data, name="zfilled_data")

    # add polarization time
    zfilled_data = np.concatenate([signal_pol, zfilled_data], axis=0)
    return zfilled_data


def gen_pol_mask(N_echoes, TE_len, polar_time, dt):
    noi_all = np.zeros(np.uint16(TE_len * N_echoes), dtype=bool)
    noi_all = add_polarization(noi_all, polar_time, dt, value=1, type=bool)
    return noi_all.astype(bool)


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
    samp_all = add_polarization(samp_all, polar_time, dt, value=pol, type=bool)
    return samp_all.astype(bool)


# %%
def sn_recognition(signal, mask, lambda_val, tol=0.1, stepsize=1, max_iter=100, method="conj_grad_l1_reg", rho=1.0):
    # mask_matrix = np.fft.fft(np.eye(len(signal)))[mask, :]
    # y = np.multiply(mask, signal)
    if method == "conj_grad_l1_reg":
        # print("ADMM with L1 regularization + Conjugate Gradient")
        mask_id = np.where(mask)
        S = selection_op(signal.shape, mask_id)
        F = ifft_op()
        A = op.composite_op(S, F)
        # x0 = np.zeros_like(signal)
        x0 = peaks_only(F.transpose(signal))
        y = S.forward(signal)
        # visualization.complex(x0, name="Initial guess")
        # visualization.complex(signal, "input signal")
        # visualization.complex(mask, "input mask")
        sn_prdct, admm_flag = admm_l2_l1(A=A, b=y, x0=x0, l1_wt=lambda_val, rho=rho,
                                       iter_max=max_iter,
                                       eps=tol)
        # sn_prdct = cg_comb(lambda_val=lambda_val, mask=mask, y=y, max_iter=max_iter, stepsize=stepsize, tol=tol).run()

        # if not admm_flag:
        #     print("Step limit reached at ADMM.")

        # visualization.plot_against_frequency(sn_prdct, len(sn_prdct), 1e-5, "EMI prediction in frequency domain")

        H = op.hadamard_op(np.abs(sn_prdct) > 1e-1 * np.max(np.abs(sn_prdct)))
        A = op.composite_op(S, F, H)
        sn_prdct, cg_flag = cg.solve_lin_cg(y, A, sn_prdct, B=op.scalar_prod_op(0.05), max_iter=1)

        # if not cg_flag:
        #     print("Step limit reached at CG for amplitude.")

        # visualization.plot_against_frequency(sn_prdct, len(sn_prdct), 1e-5, "EMI prediction in frequency domain")

        sn_prdct = F.forward(sn_prdct)

        # visualization.complex(sn_prdct, name="EMI prediction in time domain")

    else:
        print("Invalid optimization method.")

    return sn_prdct


def peaks_only(signal):
    """
    Extract peaks from the signal

    Parameters:
    - signal (numpy.ndarray): Input signal.

    Returns:
    - numpy.ndarray: Peaks.
    """
    peaks = np.zeros_like(signal)
    peaks[np.argmax(signal)] = signal[np.argmax(signal)]
    return peaks


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
