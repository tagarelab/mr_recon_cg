import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# Utility function to add a directory to sys.path if needed
import gg_lib
import optlib.c_grad as cg
import optlib.operators as op
import numpy as np
from scipy.io import savemat

# GG: First attempt at implementing cg on a simulated problem associated with epi_op

# Prep imaging vars
Sim = {}
Sim['RO_t_tot'] = 0.001  # seconds, echo DURATION!!!
Sim['PE_t_1blip'] = 0.0002
Sim['FOV'] = 0.2  # meters
Sim['B0map_filename'] = 'B0map_rand.mat'
Sim['SimRes'] = 64  # resolution of phantom (square); also size(Atrans(y))
Sim['ImgRes'] = 64 # size of data, sets acq pars, Nyquist sampling etc
if Sim['SimRes'] < Sim['ImgRes']: print("Ruhroh- don't image at higher res than your phantom definition")
Noise=100 #added to data after function call to simulate data

# Define/simulate data (requires phantom) and B0 (as needed)
phantom = shepp_logan_phantom()
Sim['SimRes']=8*Sim['SimRes'] #generate data based on cray granular phantom
phantom_resized = resize(phantom, (Sim['SimRes'], Sim['SimRes']), mode='reflect', anti_aliasing=True)
Data = gg_lib.simulatedata_Ax(Sim, phantom_resized)  # Pass Sim and Phantom as arguments
mean = 0
std_dev = Noise/ (Sim['SimRes'] ** 2)
complex_noise = np.random.normal(mean, std_dev, (Sim['ImgRes'], Sim['ImgRes'])) + 1j * np.random.normal(mean, std_dev, (Sim['ImgRes'], Sim['ImgRes']))
Data += complex_noise
Sim['SimRes']=Sim['SimRes']/8 #restore SimRes to intended recon size
#Optional line for a new random polynomial as B0 map
#gg_lib.make_b0_poly(Sim['SimRes'])  # For now, random polynomial of the right size

# quick view of what fft would look like:
plt.imshow(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(gg_lib.EPI_Reorder(Data))))), aspect='auto')
plt.title('FT(flipped EPI data)')
plt.show()
   
# Trying to plug into hemant code, looping over different SimRes
for i in [1,2,4,8,16]:
    Sim['SimRes']=i*64
    x_init=np.zeros((Sim['SimRes'], Sim['SimRes']))
    A=gg_lib.epi_op(Sim)
    x,flag=cg.c_grad(Data,A,x_init,B=op.zero_op(), max_iter=3, max_inner_iter=6, f_tol=1E-2)
    SaveFileName = f"ReconNoise{Noise}_{Sim['SimRes']}.mat"
    savemat(SaveFileName, {'x': x})
    fig, ax = plt.subplots()
    im= ax.imshow(abs(x))
    fig.colorbar(im)
    plt.title(f'CG recon res {Sim['SimRes']}')
    plt.show()
