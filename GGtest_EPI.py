# Utility function to add a directory to sys.path if needed
import sys
import gg_lib
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import scipy.io as sio
import numpy as np

# GG: Script used to test forward and backward operators for eventual use in class epi_op

# Prep imaging vars
Sim = {}
Sim['RO_t_tot'] = 0.001  # seconds, echo DURATION!!!
Sim['PE_t_1blip'] = 0.0002
Sim['FOV'] = 0.2  # meters
Sim['B0map_filename'] = 'B0map_rand.mat'
Sim['SimRes'] = 64  # resolution of phantom (square); also size(Atrans(y))
Sim['ImgRes'] = 32  # size of data, sets acq pars, Nyquist sampling etc
Sim['Noise'] = 0  # not added to code yet

if Sim['SimRes'] < Sim['ImgRes']:
    print("Ruhroh- don't image at higher res than your phantom definition")

# Simulate phantom, data, and B0 (if needed)
# Define phantom
phantom = shepp_logan_phantom()
phantom_resized = resize(phantom, (Sim['SimRes'], Sim['SimRes']), mode='reflect', anti_aliasing=True)

# Optional line for a new random polynomial
gg_lib.make_b0_poly(Sim['SimRes'])  # For now, random polynomial of the right size

# Load the SimB0map
SimB0map_data = sio.loadmat(Sim['B0map_filename'])
SimB0map = SimB0map_data['f']

# Simulate data and do EPI reordering
Data = gg_lib.simulatedata_Ax(Sim, phantom_resized)  # Pass Sim and Phantom as arguments
ReorderedData=gg_lib.EPI_Reorder(Data)

# Quick check that these operators look like transposes of each other
    # Generate random data
randx = np.random.rand(*phantom_resized.shape) + 1j* np.random.rand(*phantom_resized.shape)
randy = np.random.rand(*Data.shape) +1j*np.random.rand(*Data.shape)
    # Calculate Ax and Atrans_y
A_randx = gg_lib.simulatedata_Ax(Sim, randx)
At_randy = gg_lib.Atrans_y(Sim, randy)
     # Calculate the inner products <Ax, y> and <Atrans_y, x>
Norm1 = np.dot(np.conj(A_randx).flatten() , randy.flatten())  # <Ax, y>
Norm2 = np.dot(np.conj(randx).flatten() , At_randy.flatten())  # <Atrans_y, x>
print("Dot product 1:", Norm1)
print("Dot product 2:", Norm2)

# Create a figure with 2 rows and 3 columns of subplots
plt.figure(figsize=(18, 12))

# Plot raw EPI data
plt.subplot(2, 3, 1)
plt.imshow(np.real(Data), aspect='auto')
plt.title('raw EPI data, real')

# Plot flipped EPI data
plt.subplot(2, 3, 4)
plt.imshow(np.real(ReorderedData), aspect='auto')
plt.title('flipped EPI data, real')

# Plot FT(raw EPI data)
plt.subplot(2, 3, 2)
plt.imshow(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Data)))), aspect='auto')
plt.title('FT(raw EPI data)')

# Plot FT(flipped EPI data)
plt.subplot(2, 3, 5)
plt.imshow(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ReorderedData)))), aspect='auto')
plt.title('FT(flipped EPI data)')

# Plot reference
plt.subplot(2, 3, 3)
plt.imshow(np.real(phantom_resized), aspect='auto')
plt.title('Ref')

# Plot B0 map
plt.subplot(2, 3, 6)
plt.imshow(SimB0map, aspect='auto', vmin=-100, vmax=100)
plt.title('B0 map')
plt.colorbar()

plt.tight_layout()
plt.show()

