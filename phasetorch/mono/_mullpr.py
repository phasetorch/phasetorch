import numpy as np
from phasetorch.util._proc import MultiProc
from phasetorch.mono._mulmod import CTFTran, TIETran, MixedTran
from phasetorch.util._check import get_Ndists, assert_dims, expand_dims, assert_Ndists
from phasetorch.util._pad import calc_padwid, arr_pad, arr_unpad
from phasetorch.util._tutils import get_tensor
from phasetorch.util._utils import cphase_cproj


class MulLPR:
    def __init__(self, device, dtype, tran_func, *args):
        self.device = device
        self.dtype = dtype
        self.model = tran_func(device, dtype, *args)

    def run(self, *args):
        args = [get_tensor(v, device=self.device, dtype=self.dtype)
                for v in args]
        ph, ab = self.model(*args)
        return ph.cpu().numpy(), ab.cpu().numpy()


def ctf(rads, pix_wid, energy, prop_dists, regp=None, rel_regp=None, processes=1, device='cpu', mult_FWHM=5, dtype='float'):
    """
    Phase retrieval algorithm based on the 
    contrast transfer function (CTF) approach. 

    Parameters
    ----------
    rads : list of numpy.ndarray
        A list of numpy arrays, each containing the normalized radiograph
        measured at a given object to detector distance
    pix_wid : float
        Pixel width in mm
    energy : float
        X-ray energy in keV. Assumes monochromatic source.
    prop_dists : list
        A list of all object to detector distances.
        Must have a one-to-one correspondance with the radiographs passed using above argument.
    regp : float, default=None
        Parameter of Tikhonov regularization. 
    rel_regp : float, default=None
        Relative parameter of Tikhonov regularization.
    processes : int, default=1
        Number of parallel processes. If running on GPUs, this parameter is the total number of available GPUs.
    device : {'cpu', 'cuda'}, default='cpu'
        Target device for running this function. String 'cuda' specifies GPU execution.
    mult_FWHM : float, default=5
        Specifies the dominant width of the Fresnel impulse response function in the spatial domain as a multiple of its full width half maximum (FWHM). Used to determine padding.
    dtype : {'float', 'double'}, default='float'
        Specifies the floating point precision during computation. 'float' uses 32-bit single precision floats and 'double' uses 64-bit double precision floats.

    Returns
    -------
    delta_projs : numpy.ndarray
        A numpy array of projection of refractive index decrement 
    beta_projs : numpy.ndarray
        A numpy array of projection of absorption index
    
    Notes
    -----
    The array shape for `delta_projs` and `beta_projs` is :math:`(N_{views}, N_y, N_x)`, where :math:`N_{views}` is the total number of views, :math:`N_y` is the number of rows, and :math:`N_x` is the number of columns.
    Optionally, the shape can also be :math:`(N_y, N_x)`, which is equivalent to :math:`N_{views}=1`.
    The size of `prop_dists` is :math:`N_{dists}`, which is the total number of propagation distances. 

    """

    N_dists, prop_dists = get_Ndists(prop_dists)
    assert_dims(rads, dmin=2, dmax=4)
    # Add extra dims for views and distances if necessary
    rads = expand_dims(rads, dmax=2, dapp=0)
    rads = expand_dims(rads, dmax=3, dapp=1)
    assert_Ndists(rads, N_dists, 1)
    N_y, N_x = rads.shape[-2], rads.shape[-1]

    # Padding is computed based on Fresnel model for minimum padding. 
    # Ideally, this should be based on impulse response function used here.
    pad_y, pad_x = calc_padwid(
        N_y, N_x, pix_wid, max(prop_dists), energy, mult_FWHM)
    Npad_y = N_y + pad_y[0] + pad_y[1]
    Npad_x = N_x + pad_x[0] + pad_x[1]
    rads = arr_pad(rads, pad_y, pad_x, mode='edge')

    args = (device, dtype, CTFTran, Npad_y, Npad_x,
            pix_wid, energy, prop_dists, regp, rel_regp)
    launcher = MultiProc(MulLPR, args, processes, device)
    for i in range(rads.shape[0]):
        launcher.queue((rads[i:i+1],))
    rvals = launcher.launch_onebyone()

    phase, absorp = [], []
    for rv in rvals:
        assert len(rv) == 2
        phase.append(rv[0])
        absorp.append(rv[1])

    phase = cphase_cproj(arr_unpad(np.concatenate(
        phase, axis=0), pad_y, pad_x), energy)
    absorp = cphase_cproj(arr_unpad(np.concatenate(
        absorp, axis=0), pad_y, pad_x), energy)
    return phase, absorp


def tie(rads, pix_wid, energy, prop_dists, regp=None, rel_regp=None, processes=1, device='cuda', mult_FWHM=5, dtype='double'):
    """
    Phase retrival algorithm based on the Transport of Intensity Equation (TIE).

    Parameters
    ----------
    rads : list of numpy.ndarray
        A list of numpy arrays, each containing the normalized radiograph
        measured at a given object to detector distance
    pix_wid : float
        Pixel width in mm
    energy : float
        X-ray energy in keV. Assumes monochromatic source.
    prop_dists : list
        A list of two object to detector distances.
        Must have a one-to-one correspondance with the radiographs passed using above argument.
    regp : float, default=None
        Parameter of Tikhonov regularization. 
    rel_regp : float, default=None
        Relative parameter of Tikhonov regularization.
    processes : int, default=1
        Number of parallel processes. If running on GPUs, this parameter is the total number of available GPUs.
    device : {'cpu', 'cuda'}, default='cpu'
        Target device for running this function. String 'cuda' specifies GPU execution.
    mult_FWHM : float, default=5
        Specifies the dominant width of the Fresnel impulse response function in the spatial domain as a multiple of its full width half maximum (FWHM). Used to determine padding.
    dtype : {'float', 'double'}, default='float'
        Specifies the floating point precision during computation. 'float' uses 32-bit single precision floats and 'double' uses 64-bit double precision floats.
    
    Returns
    -------
    delta_projs : numpy.ndarray
        A numpy array of projection of refractive index decrement 
    beta_projs : numpy.ndarray
        A numpy array of projection of absorption index
    
    Notes
    -----
    The array shape for `delta_projs` and `beta_projs` is :math:`(N_{views}, N_y, N_x)`, where :math:`N_{views}` is the total number of views, :math:`N_y` is the number of rows, and :math:`N_x` is the number of columns.
    Optionally, the shape can also be :math:`(N_y, N_x)`, which is equivalent to :math:`N_{views}=1`.
    The size of `prop_dists` is :math:`N_{dists}`, which is the total number of propagation distances. 

    """

    N_dists, prop_dists = get_Ndists(prop_dists)
    assert N_dists == 2, 'Only two distances are supported'
    assert_dims(rads, dmin=4, dmax=4)
    assert_Ndists(rads, N_dists, 1)
    N_y, N_x = rads.shape[-2], rads.shape[-1]

    pad_y, pad_x = calc_padwid(
        N_y, N_x, pix_wid, max(prop_dists), energy, mult_FWHM)
    Npad_y = N_y + pad_y[0] + pad_y[1]
    Npad_x = N_x + pad_x[0] + pad_x[1]
    rads = arr_pad(rads, pad_y, pad_x, mode='edge')

    args = (device, dtype, TIETran, Npad_y, Npad_x,
            pix_wid, energy, prop_dists, regp, rel_regp)
    launcher = MultiProc(MulLPR, args, processes, device)
    for i in range(rads.shape[0]):
        launcher.queue((rads[i:i+1],))
    rvals = launcher.launch_onebyone()

    phase, absorp = [], []
    for rv in rvals:
        assert len(rv) == 2
        phase.append(rv[0])
        absorp.append(rv[1])

    phase = cphase_cproj(arr_unpad(np.concatenate(
        phase, axis=0), pad_y, pad_x), energy)
    absorp = cphase_cproj(arr_unpad(np.concatenate(
        absorp, axis=0), pad_y, pad_x), energy)
    return phase, absorp


def mixed(rads, pix_wid, energy, prop_dists, regp=None, rel_regp=None, tol=0, max_iter=3, processes=1, device='cpu', mult_FWHM=5, dtype='double'):
    """
    Phase retrieval based on the Mixed approach that combines Contrast Transfer Function (CTF) and Transport of Intensity (TIE) phase retrieval methods.

    Parameters
    ----------
    rads : list of numpy.ndarray
        A list of numpy arrays, each containing the normalized radiograph
        measured at a given object to detector distance
    pix_wid : float
        Pixel width in mm
    energy : float
        X-ray energy in keV. Assumes monochromatic source.
    prop_dists : list
        A list of two object to detector distances.
        Must have a one-to-one correspondance with the radiographs passed using above argument.
    regp : float, default=None
        Parameter of Tikhonov regularization. 
    rel_regp : float, default=None
        Relative parameter of Tikhonov regularization.
    tol : float, default=0
        Stop iterations when percentage change in reconstruction is less than tolerance.
    max_iter : default=3
        Maximum number of iterations. By default, run for 3 iterations irrespective of the tolerance ``tol``. 
    processes : int, default=1
        Number of parallel processes. If running on GPUs, this parameter is the total number of available GPUs.
    device : {'cpu', 'cuda'}, default='cpu'
        Target device for running this function. String 'cuda' specifies GPU execution.
    mult_FWHM : float, default=5
        Specifies the dominant width of the Fresnel impulse response function in the spatial domain as a multiple of its full width half maximum (FWHM). Used to determine padding.
    dtype : {'float', 'double'}, default='float'
        Specifies the floating point precision during computation. 'float' uses 32-bit single precision floats and 'double' uses 64-bit double precision floats.
    
    Returns
    -------
    delta_projs : numpy.ndarray
        A numpy array of projection of refractive index decrement 
    beta_projs : numpy.ndarray
        A numpy array of projection of absorption index
    
    Notes
    -----
    The array shape for `delta_projs` and `beta_projs` is :math:`(N_{views}, N_y, N_x)`, where :math:`N_{views}` is the total number of views, :math:`N_y` is the number of rows, and :math:`N_x` is the number of columns.
    Optionally, the shape can also be :math:`(N_y, N_x)`, which is equivalent to :math:`N_{views}=1`.
    The size of `prop_dists` is :math:`N_{dists}`, which is the total number of propagation distances. 
    
    """

    N_dists, prop_dists = get_Ndists(prop_dists)
    assert_dims(rads, dmin=2, dmax=4)
    # Add extra dims for views and distances if necessary
    rads = expand_dims(rads, dmax=2, dapp=0)
    rads = expand_dims(rads, dmax=3, dapp=1)
    assert_Ndists(rads, N_dists, 1)
    N_y, N_x = rads.shape[-2], rads.shape[-1]

    pad_y, pad_x = calc_padwid(
        N_y, N_x, pix_wid, max(prop_dists), energy, mult_FWHM)
    Npad_y = N_y + pad_y[0] + pad_y[1]
    Npad_x = N_x + pad_x[0] + pad_x[1]
    rads = arr_pad(rads, pad_y, pad_x, mode='edge')

    args = (device, dtype, MixedTran, Npad_y, Npad_x, pix_wid,
            energy, prop_dists, regp, rel_regp, tol, max_iter)
    launcher = MultiProc(MulLPR, args, processes, device)
    for i in range(rads.shape[0]):
        launcher.queue((rads[i:i+1],))
    rvals = launcher.launch_onebyone()

    phase, absorp = [], []
    for rv in rvals:
        assert len(rv) == 2
        phase.append(rv[0])
        absorp.append(rv[1])

    phase = cphase_cproj(arr_unpad(np.concatenate(
        phase, axis=0), pad_y, pad_x), energy)
    absorp = cphase_cproj(arr_unpad(np.concatenate(
        absorp, axis=0), pad_y, pad_x), energy)

    return phase, absorp
