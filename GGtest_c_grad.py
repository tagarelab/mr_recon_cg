import numpy as np
import optlib.operators as op
import optlib.c_grad as cg

# Utility function to add a directory to sys.path if needed
import sys
import gg_lib
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import scipy.io as sio

# GG: First attempt at implementing cg on a simulated problem associated with epi_op

# Prep imaging vars
Sim = {}
Sim['RO_t_tot'] = 0.001  # seconds, echo DURATION!!!
Sim['PE_t_1blip'] = 0.0002
Sim['FOV'] = 0.2  # meters
Sim['B0map_filename'] = 'B0map_rand.mat'
Sim['SimRes'] = 32  # resolution of phantom (square); also size(Atrans(y))
Sim['ImgRes'] = Sim['SimRes']  # size of data, sets acq pars, Nyquist sampling etc
Sim['Noise'] = 0  # not added to code yet
if Sim['SimRes'] < Sim['ImgRes']: print("Ruhroh- don't image at higher res than your phantom definition")
#Optional line for a new random polynomial
#gg_lib.make_b0_poly(Sim['SimRes'])  # For now, random polynomial of the right size

# Simulate/define phantom, data, and B0 (as needed)
phantom = shepp_logan_phantom()
phantom_resized = resize(phantom, (Sim['SimRes'], Sim['SimRes']), mode='reflect', anti_aliasing=True)
Data = gg_lib.simulatedata_Ax(Sim, phantom_resized)  # Pass Sim and Phantom as arguments
x_init=0*phantom_resized

# Trying to plug into hemant code
A=gg_lib.epi_op(Sim)
x,flag=cg.c_grad(Data,A,x_init,B=op.zero_op(), max_iter=3, max_inner_iter=6, f_tol=1e-5)

