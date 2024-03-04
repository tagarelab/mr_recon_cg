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


def scatter3d(B0_SI, B0_LR, B0_AP, grad, mask=None):
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
    X_M, Y_M, Z_M = np.meshgrid(B0_SI, B0_LR, B0_AP, indexing='ij')

    if mask is not None:
        X_M = X_M[mask]
        Y_M = Y_M[mask]
        Z_M = Z_M[mask]
        grad = grad[mask]

    scatter = ax.scatter(X_M, Y_M, Z_M, c=grad, s=1)
    plt.colorbar(scatter)

    # ax.set_title("Liver Gradient at "+grad_str+" mT/m")
    ax.set_xlabel('SI (m)')
    ax.set_ylabel('LR (m)')
    ax.set_zlabel('AP (m)')
    ax.axis('equal')

    plt.show()
