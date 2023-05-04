import numpy as np
from phasetorch.util import path_lengths_sphere 
from phasetorch.mono.sim import simdata, simdata_CTF, simdata_TIE
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
import os
import yaml
import glob
import numpy as np

def read_yaml_params(yaml_file): 
    files = glob.glob(yaml_file)

    params = []
    for fl in files:
        with open(fl, "r") as fid:
            dt = yaml.safe_load(fid)
        
        params.append(dt)

    return params

def sim_rads(num_z, num_x, pix_width, energy, prop_dists, sim_upsmpl, processes, device, radii, center_y, center_x, delta_gt, beta_gt, forw_model="FP", **kwargs):
    #kwargs collects unused arguments

    nup_z, nup_x = sim_upsmpl*num_z, sim_upsmpl*num_x
    pup_wid = pix_width/sim_upsmpl

    # Initialize the projections of delta and beta to zeros.
    pdelta_gt = np.zeros((nup_z, nup_x), dtype=np.float32)
    pbeta_gt = np.zeros((nup_z, nup_x), dtype=np.float32)

    # Mask denotes the region with non-zero projection values. It is used later for background subtraction.
    pmask_gt = np.zeros((nup_z, nup_x), dtype=np.float32)

    # Total projection is the sum of projections through all the spheres.
    # We assume that the spheres are non-overlapping along the X-ray propagation axis.
    for dl, bt, cy, cx, rd in zip(delta_gt, beta_gt, center_y, center_x, radii):
        path, mask = path_lengths_sphere(nup_z, nup_x, pup_wid, cy, cx, rd)
        pdelta_gt += dl*path
        pbeta_gt += bt*path
        pmask_gt += mask

    # Add one unitary dimension to indicate one view. "gt" indicates ground-truth.
    pdelta_gt, pbeta_gt = pdelta_gt[np.newaxis], pbeta_gt[np.newaxis]

    # Simulation of X-ray transmission images.
    if forw_model == "FP":
        rads = simdata(pdelta_gt, pbeta_gt, pup_wid, energy, prop_dists, processes=processes, device=device)
    elif forw_model == "CTF":
        rads = simdata_CTF(pdelta_gt, pbeta_gt, pup_wid, energy, prop_dists, processes=processes, device=device)
    elif forw_model == "TIE":
        rads = simdata_TIE(pdelta_gt, pbeta_gt, pup_wid, energy, prop_dists, processes=processes, device=device)
    else:
        raise("ValueError: Forward model not recognized.")
    
    # Downsample the X-ray images.
    rads = downscale_local_mean(rads, factors=(1, 1, sim_upsmpl, sim_upsmpl))

    # Downsample the projections for comparison with phase-retrieved images.
    pdelta_gt = downscale_local_mean(pdelta_gt, factors=(1, sim_upsmpl, sim_upsmpl))
    pbeta_gt = downscale_local_mean(pbeta_gt, factors=(1, sim_upsmpl, sim_upsmpl))

    # Downsample the mask.
    pmask_gt = downscale_local_mean(pmask_gt, factors=(sim_upsmpl, sim_upsmpl))
    pmask_gt = pmask_gt > 0
    pmask_gt = pmask_gt[np.newaxis]

    return rads, pdelta_gt, pbeta_gt, pmask_gt

def save_data(save_dir, rads, pdelta_gt, pdelta_est, nrmse, bground=None):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "data.npz"),
                rads=rads, pdelta_gt=pdelta_gt, pdelta_est=pdelta_est, nrmse=nrmse, bground=bground)

    for i in range(rads.shape[0]):
        for j in range(rads.shape[1]):
            plt.imshow(rads[i, j])
            plt.colorbar()
            plt.savefig(os.path.join(save_dir, "rads_{}_{}.png".format(i, j)))
            plt.close()
        
    for i in range(pdelta_gt.shape[0]):
        plt.imshow(pdelta_gt[i])
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, "pdelta_gt_{}.png".format(i)))
        plt.close()
    
    for i in range(pdelta_est.shape[0]):
        plt.imshow(pdelta_est[i])
        plt.title("NRMSE: {:.2e}".format(nrmse))
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, "pdelta_est_{}.png".format(i)))
        plt.close()
