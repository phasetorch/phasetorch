import numpy as np
from phasetorch.mono._simtor import simdata

def evalfit(rads, pix_wid, energy, prop_dists, delta_projs, beta_projs, processes=1, device='cpu', mult_FWHM=5, dtype='double'):
    rads_sqroot = simdata(delta_projs, beta_projs, pix_wid, energy, prop_dists, processes=processes, device=device, dtype=dtype, mult_FWHM=mult_FWHM, ret_sqroot=True, pad=True)
    
    rads_diff = rads - rads_sqroot
    axis = tuple([i for i in range(rads_diff.ndim) if i != 0])
    losses = np.mean(rads_diff**2, axis=axis)

    return losses, rads_diff

