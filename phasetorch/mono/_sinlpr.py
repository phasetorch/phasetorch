import numpy as np
from phasetorch.mono._sinmod import PaganinTran
from phasetorch.util._check import assert_dims,assert_Ndists,expand_dims
from phasetorch.util._proc import MultiProc
from phasetorch.util._tutils import get_tensor
from phasetorch.util._pad import arr_pad,arr_unpad,calc_padwid
from phasetorch.util._utils import cphase_cproj

class SinLPR:
    def __init__(self, device, dtype, *args):
        self.device = device
        self.dtype = dtype
        self.model = PaganinTran(device, dtype, *args)

    def run(self, *args):
        args = [get_tensor(v, device=self.device, dtype=self.dtype) for v in args]
        return self.model(*args).detach().cpu().numpy()

def paganin(rads, pix_wid, energy, prop_dist, delta_over_beta, processes=1, device='cpu', mult_FWHM=5, mag_fact=1.0, dtype='float'):
    """Linear phase retrieval algorithm for imaging of single-material objects using single propagation distance phase contrast CT that is also popularly known as Paganin's phase retrieval.

    Retrieves the refractive index decrement projections from normalized radiographs acquired at a single propagation distance. These projections are proportional to the negative logarithm of a low-pass filtered version of the input radiographs. As input, it takes normalized radiographs that are acquired at a single object to detector propagation distance and multiple views. It returns the projections of the refractive index decrement using Paganin's phase retrieval. As additional inputs, this function also requires specification of the pixel width, X-ray energy, propagation distance, and the ratio of delta over beta. The same length units should be used for all input parameters.

    Parameters
    ----------
    rads : numpy.ndarray
        Normalized radiographs.
    pix_wid : float
        Pixel width.
    energy : float
        X-ray energy.
    prop_dist : float
        Propagation distance from object to detector (or scintillator).
    delta_over_beta : float
        Ratio of delta over beta.
    processes : int, default=1
        Number of parallel processes. If running on GPUs, this parameter is the total number of available GPUs.
    device : {'cpu', 'cuda'}, default='cpu'
        Target device for running this function. String 'cuda' specifies GPU execution.
    mult_FWHM : float, default=5
        Specifies the dominant width of the Fresnel impulse response function in the spatial domain as a multiple of its full width half maximum (FWHM).
    mag_fact : float, default=1
        Magnification factor given by the ratio of the source to detector distance over the source to object distance. This parameter is one for parallel beam X-rays.
    dtype : {'float', 'double'}, default='float'
        Specifies the floating point precision during computation. Must be either 'float' or 'double'. 'float' uses 32-bit single precision floats and 'double' uses 64-bit double precision floats.
    
    Returns
    -------
    delta_projs : numpy.ndarray
        Refractive index decrement projections.
    
    Notes
    -----
    The array shape for `rads` and `delta_projs` is :math:`(N_{views}, N_y, N_x)`.

    Examples
    --------
    .. code-block:: python

        if __name__ == '__main__': #Required for parallel compute
            import numpy as np
            import matplotlib.pyplot as plt
            from phasetorch.sim import simdata
            from phasetorch.pr import paganin

            x, y = np.meshgrid(np.arange(0, 128.0)*0.001, np.arange(0, 128.0)*0.001) #Grid shape 128 x 128. Pixel 0.001mm.
            projs = np.zeros_like(x)[np.newaxis] #Ground-truth projections. Unitary dimension indicates one view.
            mask = (0.032**2 - (x-0.064)**2) > 0 #Mask for non-zero projections of 0.032mm cylinder.
            projs[0][mask] = 2*np.sqrt((0.032**2 - (x-0.064)**2)[mask]) #Generate path lengths.
            delta_projs, beta_projs = projs*1e-7, projs*1e-8 #delta = 1e-7, beta=1e-8
         
            rads = simdata(delta_projs, beta_projs, 0.001, 20, 100, processes=1, device='cuda') #Radiographs at distances of 100mm and 200mm for 20keV X-rays
            rads = rads.squeeze(1) #Remove the distance dimension

            delta_projs_est = paganin(rads, 0.001, 20, 100, 10, processes=1, device='cuda') #Paganin phase retrieval

            rmse = np.sqrt(np.mean((delta_projs_est - delta_projs)**2)) #Root mean squared error
            print('Percentage RMSE is {}'.format(100*rmse/np.mean(delta_projs)))

            plt.imshow(delta_projs_est[0]) #Estimated projections of delta
            plt.show()
    """
    
    #Convert to floats
    pix_wid = float(pix_wid)
    prop_dist = float(prop_dist)
    energy = float(energy)
    delta_over_beta = float(delta_over_beta)

    if rads.ndim >= 4:
        rads = rads.squeeze(1) # Remove unitary distance dimension if it exists    
    
    assert_dims(rads, dmin=2, dmax=3)
    # Add a view dimension if max dims is 2
    rads = expand_dims(rads, dmax=2, dapp=0)
    N_y,N_x = rads.shape[-2],rads.shape[-1] 

    # Pad assuming the Fresnel phase-contrast model to minimize filter width.
    # Ideally, should be redone for the Paganin filter. 
    pad_y,pad_x = calc_padwid(N_y, N_x, pix_wid, prop_dist, energy, mult_FWHM)
    Npad_y = N_y + pad_y[0] + pad_y[1]
    Npad_x = N_x + pad_x[0] + pad_x[1]
    rads = arr_pad(rads, pad_y, pad_x, mode='edge')
 
    args = (device, dtype, Npad_y, Npad_x, pix_wid, energy, prop_dist, mag_fact) 
    launcher = MultiProc(SinLPR, args, processes, device)                 
    for i in range(rads.shape[0]):
        launcher.queue((rads[i:i+1], delta_over_beta))
    rvals = launcher.launch_onebyone()
    projs = np.concatenate(rvals, axis=0)
    projs = cphase_cproj(projs, energy)
    
    delta_projs = arr_unpad(projs, pad_y, pad_x)
    return delta_projs
