import numpy as np
import torch
from phasetorch.util._utils import get_wavelength
from phasetorch.util._tutils import get_tensor
from phasetorch.mono._fresmod import FresnelProp
from phasetorch.mono._mulmod import CTFTran, TIETran
from phasetorch.util._pad import calc_padwid, arr_pad, arr_unpad, next_fft_len
from phasetorch.util._check import get_Ndists,assert_dims,expand_dims
from phasetorch.util._proc import MultiProc

# Mono-energetic case
class SimFP:
    def __init__(self, device, dtype, *args):
    # VS: args here refers to the parameters : (Npad_y, Npad_x, pix_wid, energy, prop_dists)
        self.device = device
        self.dtype = dtype
        self.model = FresnelProp(device, dtype, *args).to(device)

    # VS: The *args here is different from that in __init__()
    #     Here args is a tuple consisting a single input 5D numpy-array of size (#Nviews, #Ndists=1, #rows, #cols, 2),  ...
    #     where Nviews is typically set to 1, but can be more than 1 in general
    def run(self, *args):
        args = [get_tensor(v, device=self.device, dtype=self.dtype) for v in args]
        with torch.no_grad():
            return self.model(*args).detach().cpu().numpy() # Square root data
            # VS: I don't think detach() is required since requires_grad=False for all tensors in the computation.

class SimCTF:
    def __init__(self, device, dtype, *args):
    # VS: args here refers to the parameters : (Npad_y, Npad_x, pix_wid, energy, prop_dists, reg_par=None, rel_reg_par=None)
        self.device = device
        self.dtype = dtype
        self.model =  CTFTran(device, dtype, *args)

    # VS: The *args here is different from that in __init__()
    #     Here args is a tuple consisting of two 3D numpy-arrays, each of size (#Nviews, #rows, #cols),  ...
    #     where Nviews is typically set to 1, but can be more than 1 in general
    def run(self, *args):
        with torch.no_grad():
            return self.model.prop(*args).detach().cpu().numpy() # Square root data
        # VS: I don't think detach() is required since requires_grad=False for all tensors in the computation.

class SimTIE:
    def __init__(self, device, dtype, *args):
    # VS: args here refers to the parameters : (Npad_y, Npad_x, pix_wid, energy, prop_dists, reg_par=None, rel_reg_par=None)
        self.device = device
        self.dtype = dtype
        self.model =  TIETran(device, dtype, *args)

    # VS: The *args here is different from that in __init__()
    #     Here args is a tuple consisting of two 3D numpy-arrays, each of size (#Nviews, #rows, #cols),  ...
    #     where Nviews is typically set to 1, but can be more than 1 in general
    def run(self, *args):
        with torch.no_grad():
            return self.model.prop(*args).detach().cpu().numpy() # Square root data
        # VS: I don't think detach() is required since requires_grad=False for all tensors in the computation.


def cmplxtrans(delta_projs, beta_projs, energy):
    wavelength = get_wavelength(energy)
    trans = np.exp(-1j*2*np.pi*(delta_projs-1j*beta_projs)/wavelength)
    return trans

# Mono-energetic case.
def simdata(delta_projs, beta_projs, pix_wid, energy, prop_dists, processes=1, device='cpu', dtype='float', mult_FWHM=5, ret_sqroot=False, pad=True):
    """
    Simulate normalized radiographs that contain both absorption and phase contrast from projections of absorption index and refractive index decrement.

    Simulates normalized radiographs (pixel values that range from 0 to 1 approx.) for X-ray phase contrast CT.
    As input, it takes the projections of the refractive index decrement (delta), :math:`{\delta}`, and absorption index (beta), :math:`{\\beta}`, at various views and object to detector propagation distances.
    To simulate phase and absorption contrast in the radiographs, the function also needs the pixel width, X-ray energy, and propagation distances as additional inputs.
    The returned array is the normalized radiographs at all the views and propagation distances.
    The same length units should be used for all input parameters.

    Parameters
    ----------
    delta_projs : numpy.ndarray
        Refractive index decrement projections.
    beta_projs : numpy.ndarray
        Absorption index projections.
    pix_wid : float
        Pixel width.
    energy : float
        X-ray energy (in keV).
    prop_dists : list of float
        Propagation distances from object to detector (or scintillator).
    processes : int, default=1
        Number of parallel processes. If running on GPUs, this parameter is the total number of available GPUs.
    device : {'cpu', 'cuda'}, default='cpu'
        Target device for running this function. String 'cuda' specifies GPU execution.
    dtype : {'float', 'double'}, default='float'
        Specifies the floating point precision during computation. 'float' uses 32-bit single precision floats and 'double' uses 64-bit double precision floats.
    mult_FWHM : float, default=5
        Specifies the dominant width of the Fresnel impulse response function in the spatial domain as a multiple of its full width half maximum (FWHM). Used to determine padding.
    ret_sqroot: bool, detault=False
        If True, return the square root of the X-ray intensity measurements i.e., square root of the measured radiographs.
    pad : bool, default=True
        If True, use appropriate amount of padding for the projection arrays during simulation.

    Returns
    -------
    rads : numpy.ndarray
        Simulated normalized radiographs.

    Notes
    -----
    The array shape for `delta_projs` and `beta_projs` is :math:`(N_{views}, N_y, N_x)`, where :math:`N_{views}` is the total number of views, :math:`N_y` is the number of rows, and :math:`N_x` is the number of columns.
    Optionally, the shape can also be :math:`(N_y, N_x)`, which is equivalent to :math:`N_{views}=1`.
    The size of `prop_dists` is :math:`N_{dists}`, which is the total number of propagation distances.
    The returned normalized radiographs `rads` is of shape :math:`(N_{views}, N_{dists}, N_y, N_x)`.

    Examples
    --------
    .. code-block:: python

        if __name__ == '__main__': #Required for parallel compute
            import numpy as np
            import matplotlib.pyplot as plt
            from phasetorch.sim import simdata

            x, y = np.meshgrid(np.arange(0, 128.0)*0.001, np.arange(0, 128.0)*0.001) #Grid shape 128 x 128. Pixel 0.001mm.
            projs = np.zeros_like(x)[np.newaxis] #Ground-truth projections. Unitary dimension indicates one view.
            mask = (0.032**2 - (x-0.064)**2) > 0 #Mask for non-zero projections of 0.032mm cylinder.
            projs[0][mask] = 2*np.sqrt((0.032**2 - (x-0.064)**2)[mask]) #Generate path lengths.
            delta_projs, beta_projs = projs*1e-7, projs*1e-8 #delta = 1e-7, beta=1e-8

            rads = simdata(delta_projs, beta_projs, 0.001, 20, [100, 200], processes=1, device='cuda') #Radiographs at distances of 100mm and 200mm for 20keV X-rays

            plt.imshow(rads[0, 0]) #Radiograph at 100mm
            plt.show()

            plt.imshow(rads[0, 1]) #Radiograph at 200mm
            plt.show()
    """

    #Convert to floats
    pix_wid = float(pix_wid)
    energy = float(energy)

    N_y,N_x = delta_projs.shape[-2],delta_projs.shape[-1]
    assert delta_projs.shape == beta_projs.shape
    N_dists,prop_dists = get_Ndists(prop_dists)

    trans = cmplxtrans(delta_projs, beta_projs, energy)
    assert_dims(trans, dmin=2, dmax=3)
    # VS: data dimensions - (#views, #dists, #detector-rows, #detector-columns).
    trans = expand_dims(trans, dmax=2, dapp=0)
    trans = expand_dims(trans, dmax=3, dapp=1)

    # VS: pad data based on chirp-function Nquist sampling criteria. refer Voelz and Rogemann paper, 2009.
    if pad:
        pad_y, pad_x = calc_padwid(N_y, N_x, pix_wid, max(prop_dists), energy, mult_FWHM)
        Npad_y = N_y + pad_y[0] + pad_y[1]
        Npad_x = N_x + pad_x[0] + pad_x[1]
        trans = arr_pad(trans, pad_y, pad_x, mode='edge')
    else:
        Npad_y, Npad_x = N_y, N_x

    #VS: data dimensions - (#views, #dists=1, #detector-rows, #detector-columns, 2) where the last dimension is Re/Im part
    trans = np.stack((np.real(trans), np.imag(trans)), axis=-1)
    assert_dims(trans, dmin=3, dmax=5, dlast=2)

    args = (device, dtype, Npad_y, Npad_x, pix_wid, energy, prop_dists)
    # VS: init_args (parameters) for the model to be executed by launcher, excludes input data to the forward model that is specified separately
    launcher = MultiProc(SimFP, args, processes, device)
    # For every view, apply fresnel propagator to the corresponding transmission field.
    # the fresnel integral is computed for multiple detector-distances at a time.
    # the input to the fresnel propagator is a 5D tensor of size (#views=1, #dists, #rows, #columns, 2)
    for i in range(trans.shape[0]):
        launcher.queue((trans[i:i+1],))

    data = launcher.launch_onebyone()
    data = np.concatenate(data, axis=0)
    # after concatenation data is of size (#views, #dists, Ny, Nx)
    if pad:
        data = arr_unpad(data, pad_y, pad_x)

    rads = data #.squeeze()
    return rads if ret_sqroot else rads**2

# Mono-energetic case: Simulate data using the CTF approxmimation to the Fresnel model
def simdata_CTF(delta_projs, beta_projs, pix_wid, energy, prop_dists, processes=1, device='cpu', dtype='float', mult_FWHM=5, pad=True):
    #Convert to floats
    pix_wid = float(pix_wid)
    energy = float(energy)
    # beta_projs / delta_projs is a 3D numpy array of (#views, Ny, Nx)
    assert (beta_projs.ndim==3) and (delta_projs.ndim==3) and (beta_projs.shape == delta_projs.shape)
    N_y,N_x = delta_projs.shape[-2],delta_projs.shape[-1]
    N_dists,prop_dists = get_Ndists(prop_dists)

    # VS: pad data based on chirp-function Nquist sampling criteria. refer Voelz and Rogemann paper, 2009.
    # VS: In both simdata and simdata_CTF, we assume that the X-ray energy (keV) is sufficiently high, so that ...
    #     sampling the fresnel transform (or its approximation) in the fourier domain is permitted (without leading to aliasing).
    #     There are (isolated) cases where this may not be true.
    if pad:
        pad_y, pad_x = calc_padwid(N_y, N_x, pix_wid, max(prop_dists), energy, mult_FWHM)
        Npad_y = N_y + pad_y[0] + pad_y[1]
        Npad_x = N_x + pad_x[0] + pad_x[1]
        beta_projs  = arr_pad(beta_projs, pad_y, pad_x, mode='edge')
        delta_projs = arr_pad(delta_projs, pad_y, pad_x, mode='edge')
    else:
        Npad_y, Npad_x = N_y, N_x

    #VS: The last 2 args pertaining to optional scalar regularization parameters are unused
    args = (device, dtype, Npad_y, Npad_x, pix_wid, energy, prop_dists, None, None)
    # VS: init_args (parameters) for the model to be executed by launcher, excludes input data to the forward model that is specified separately
    launcher = MultiProc(SimCTF, args, processes, device)
    # For every view, apply CTF propagator to the corresponding transmission field.
    # the CTF propagator is applied for multiple detector-distances at a time.
    for i in range(beta_projs.shape[0]):
        launcher.queue((beta_projs[i:i+1], delta_projs[i:i+1]))

    data = launcher.launch_onebyone()
    data = np.concatenate(data, axis=0)
    # after concatenation data is of size (#views, #dists, Ny, Nx)
    if pad:
        data = arr_unpad(data, pad_y, pad_x)

    if np.any(data < 0):
        print("WARNING: Simulation contains negative values. This may occur if the projections don't satisfy the CTF approximations.")
    # The above computed radiographs are intensity radiographs
    return data #.squeeze()

# Mono-energetic case: Simulate data using the TIE approxmimation to the Fresnel model
def simdata_TIE(delta_projs, beta_projs, pix_wid, energy, prop_dists, processes=1, device='cpu', dtype='float', second_order=False):
    #Convert to floats
    pix_wid = float(pix_wid)
    energy = float(energy)
    # beta_projs / delta_projs is a 3D numpy array of (#views, Ny, Nx)
    assert (beta_projs.ndim==3) and (delta_projs.ndim==3) and (beta_projs.shape == delta_projs.shape)
    N_y,N_x = delta_projs.shape[-2],delta_projs.shape[-1]
    N_dists,prop_dists = get_Ndists(prop_dists)

    #VS: The last 2 args pertaining to optional scalar regularization parameters are unused, one of them must be set non-zero to avoid warnings
    args = (device, dtype, N_y, N_x, pix_wid, energy, prop_dists, None, 1e-06)
    # VS: init_args (parameters) for the model to be executed by launcher, excludes input data to the forward model that is specified separately
    launcher = MultiProc(SimTIE, args, processes, device)
    # For every view, apply TIE propagator to the corresponding transmission field.
    # the TIE propagator is applied for multiple detector-distances at a time.
    for i in range(beta_projs.shape[0]):
        launcher.queue((beta_projs[i:i+1], delta_projs[i:i+1], second_order))

    data = launcher.launch_onebyone()
    data = np.concatenate(data, axis=0)
    if np.any(data < 0):
        print("WARNING: Simulation contains negative values. This may occur if the projections don't satisfy the TIE approximations.")

    # The above computed radiographs are intensity radiographs
    return data #.squeeze()
