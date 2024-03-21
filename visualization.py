"""
   Name: visualization.py
   Purpose:
   Created on: 2/22/2024
   Created by: Heng Sun
   Additional Notes: 
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


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


def scatter3d(B0_LR, B0_SI, B0_AP, grad, xlim=None, ylim=None, zlim=None, clim=None, mask=None, title=None):
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
    ax.set_xlabel('LR (mm)')
    ax.set_ylabel('SI (mm)')
    ax.set_zlabel('AP (mn)')
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
