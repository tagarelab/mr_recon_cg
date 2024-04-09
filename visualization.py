"""
   Name: visualization.py
   Purpose:
   Created on: 2/22/2024
   Created by: Heng Sun
   Additional Notes: 
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import fft, optimize
import algebra

__all__ = ['quiver3d', 'scatter3d']


def quiver3d(vector, orig=None, label=None, xlim=None, ylim=None, zlim=None, title=None):
    """
    3D quiver plot
    :param title:
    :return:
    """
    if orig is None:
        orig = [0, 0, 0]

    if label is None:
        label = np.arange(vector.shape[1])

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    ax = plt.figure().add_subplot(projection='3d')

    for i in range(vector.shape[1]):
        ax.quiver(orig[0], orig[1], orig[2], vector[0, i], vector[1, i], vector[2, i], label=label[i],
                  colors=colors[i % len(colors)])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axis('equal')

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if zlim is not None:
        ax.set_zlim(zlim)

    ax.legend()
    plt.show()


def scatter3d(B0_LR, B0_SI, B0_AP, grad, xlim=None, ylim=None, zlim=None, clim=None, mask=None, title=None,
              xlabel="LR (mm)", ylabel="SI (mm)", zlabel="AP (mm)"):
    """
    3D scatter plot
    This function is adapted from a MATLAB function by Github Copilot and edited & tested by the author.

    Parameters:
    - B0_SI (numpy.ndarray): The B0 field map in the SI direction.
    - B0_LR (numpy.ndarray): The B0 field map in the LR direction.
    - B0_AP (numpy.ndarray): The B0 field map in the AP direction.
    - grad (numpy.ndarray): The gradient.
    - grad_str (str): The gradient string.

    Returns:
    - None
    """

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    X_M, Y_M, Z_M = np.meshgrid(B0_LR, B0_SI, B0_AP, indexing='ij')

    if mask is not None:
        X_M = X_M[mask]
        Y_M = Y_M[mask]
        Z_M = Z_M[mask]
        if grad.ndim == 3:
            grad = grad[mask]

    scatter = ax.scatter(X_M, Y_M, Z_M, c=grad, s=1)

    plt.colorbar(scatter)

    # ax.set_title("Liver Gradient at "+grad_str+" mT/m")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.axis('equal')

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if zlim is not None:
        ax.set_zlim(zlim)

    if clim is not None:
        scatter.set_clim(clim[0], clim[1])

    plt.show()


def sig_time(time, signal):
    """
    2D signal plot
    :param signal: signal
    :param time: time
    :return: None
    """
    plt.figure()
    plt.plot(time, np.abs(signal), label='Magnitude')
    plt.plot(time, np.real(signal), label='Real')
    plt.plot(time, np.imag(signal), label='Imaginary')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.show()


def imshow(image, name=None):
    plt.imshow(image, cmap='gray')
    if name is not None:
        plt.title(name)
    plt.colorbar()
    plt.show()


def snr_tradeoff_compare(sig_1, sig_2, noi_range=None, sig_1_name="Signal 1", sig_2_name="Signal 2"):
    N_rep = sig_1.shape[1]
    N_sig_ch = sig_1.shape[2]
    snr_1 = np.zeros((N_rep, N_sig_ch))
    snr_2 = np.zeros((N_rep, N_sig_ch))

    for i in range(N_rep):
        for k in range(N_sig_ch):
            snr_1[i, k] = algebra.snr(sig_1[:, :i, k], noi_range)
            snr_2[i, k] = algebra.snr(sig_2[:, :i, k], noi_range)

    if N_rep == 1:
        print('Only one repetition is entered. SNR tradeoff not plotted.')
    else:
        xaxis_temp = np.arange(N_rep) + 1
        for k in range(N_sig_ch):
            plt.plot(xaxis_temp, snr_1[:, k], label=sig_1_name)
            plt.plot(xaxis_temp, snr_2[:, k], label=sig_2_name)
            plt.xlabel('# of Scans Averaged')
            plt.ylabel('SNR')
            plt.title('SNR trade-off with # of Averages in ch' + str(k))
            plt.legend()
            plt.show()

    return snr_1, snr_2


def curve_fit(data, func, name=None, xaxis=None, p0=None):
    # got this function from stack overflow
    if xaxis is None:
        xaxis = np.arange(len(data))

    if p0 is not None:
        fit = optimize.curve_fit(func, xaxis, data, p0=p0)
    else:
        fit = optimize.curve_fit(func, xaxis, data)

    # recreate the fitted curve using the optimized parameters
    data_fit = func(xaxis, *fit[0])

    if name is not None:
        plt.title(name)

    reminder_std = np.std(np.abs(data - data_fit))
    plt.ylabel('std = {}'.format(reminder_std))

    plt.plot(xaxis, data, label='data')
    plt.plot(xaxis, data_fit, label='after fitting')
    plt.legend()
    plt.show()

    return fit


def scatter_mean_std(data, name=None, xaxis=None):
    # data = abs(data)
    if xaxis is None:
        xaxis = np.arange(len(data))
    plt.plot(xaxis, data)

    mean = np.mean(data)
    std = np.std(data)

    plt.hlines(mean, xaxis[0], xaxis[-1], label='Mean = {}'.format(mean), colors=['r'])
    plt.hlines(mean + std, xaxis[0], xaxis[-1], label='1st Std = {}'.format(std),
               colors=['g'])
    plt.hlines(mean - std, xaxis[0], xaxis[-1], colors=['g'])

    if name is not None:
        plt.title(name)

    plt.legend()
    plt.show()

    return mean, std


def plot_against_frequency(signal, frag_len, dt, name=None, ylim=None):
    freq_axis = fft.fftshift(fft.fftfreq(frag_len, dt)) / 1000
    plt.plot(freq_axis, abs(signal), label='Magnitude')
    plt.plot(freq_axis, np.real(signal), label='Real')
    plt.plot(freq_axis, np.imag(signal), label='Imaginary')
    plt.legend()
    plt.xlabel('Frequency (kHz)')
    if ylim is not None:
        plt.ylim(ylim)
    if name is not None:
        plt.title(name)
    plt.show()


def freq_plot(signal, dt, name=None, ylim=None):
    signal_ft = fft.fftshift(fft.fft(fft.fftshift(signal)))
    length = len(signal_ft)
    plot_against_frequency(signal_ft, length, dt, name, ylim)


def repetitions(signal, name=None):
    im = plt.imshow(abs(signal), cmap=cm.coolwarm, interpolation='nearest', vmin=0, vmax=0.5e6,
                    aspect='auto')
    plt.colorbar(im)

    if name is not None:
        plt.title(name)

    plt.show()


def freq_analysis(signal, frag_len, dt, name=None, type='heatmap'):
    signal = signal.reshape(-1)
    N_frag = int(len(signal) / frag_len)
    freq_mat = np.zeros((N_frag, frag_len), dtype='complex')
    for i in range(N_frag):
        freq_mat[i, :] = fft.fftshift(fft.fft(fft.fftshift(signal[i * frag_len:(i + 1) * frag_len])))

    sample_freq = 1 / dt
    step = sample_freq / frag_len
    # freq_axis = np.arange(-sample_freq / 2, sample_freq / 2, step)
    freq_axis = fft.fftshift(fft.fftfreq(frag_len, dt))
    num_echo_axis = range(N_frag)
    freq_axis, num_echo_axis = np.meshgrid(freq_axis, num_echo_axis)

    if type == '3d':
        ax = plt.axes(projection='3d')
        # ax.set_zlim([0, 30000])
        ax.set_xlabel('Freq offset (Hz)')
        ax.set_ylabel('Echoes (#)')
        surf = ax.plot_surface(freq_axis, num_echo_axis, abs(freq_mat), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

    if type == 'heatmap':
        im = plt.imshow(abs(freq_mat), cmap=cm.coolwarm, interpolation='nearest',
                        extent=[-sample_freq / 2 / 1000, sample_freq / 2 / 1000, 1, N_frag], vmin=0, vmax=0.5e6,
                        aspect='auto')
        plt.xlabel('Freq offset (kHz)')
        plt.ylabel('Echoes (#)')
        plt.colorbar(im)

    if name is not None:
        plt.title(name)

    plt.show()


def absolute(signal, name=None, ylim=None):
    plt.plot(np.abs(signal), label="abs")
    if ylim is not None:
        plt.ylim(ylim)
    if name is not None:
        plt.title(name)
    plt.legend()
    plt.show()


def complex(signal, name=None, rect=True, ylim=None, xlabel=None, ylabel=None):
    if rect is True:
        plt.plot(np.abs(signal), label="abs")
        plt.plot(np.real(signal), label="real")
        plt.plot(np.imag(signal), label="imag")
    else:
        plt.plot(np.unwrap(np.angle(signal)), label="angle")
        # plt.plot(np.angle(signal), label="angle")

    if ylim is not None:
        plt.ylim(ylim)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if name is not None:
        plt.title(name)
    plt.legend()
    plt.show()
