import pytest
from test_mono.utils import sim_rads, save_data, read_yaml_params
from phasetorch.mono.pr import ctf, tie, mixed, nlpr
from skimage.metrics import normalized_root_mse
import matplotlib.pyplot as plt
import inspect
import os
import numpy as np
from skimage.morphology import binary_erosion

def nrmse_bak(pdelta_est, pdelta_gt, pmask_gt):
    # Create mask for the background region.
    pmask_gt_bg = np.bitwise_not(pmask_gt)
    # Remove the region that are close to the image edges.
    sh = pmask_gt_bg.shape
    for i in range(pmask_gt_bg.shape[0]):
        pmask_gt_bg[i] = binary_erosion(pmask_gt_bg[i], footprint=np.ones((sh[-2]//4, sh[-1]//4))) 

#    plt.imshow(pmask_gt_bg)
#    plt.colorbar()
#    plt.show()

    bak_val = np.mean(pdelta_est[pmask_gt_bg])
    nrmse = normalized_root_mse(pdelta_est-bak_val, pdelta_gt, normalization="euclidean")    
    return nrmse, bak_val

def eval_ctf(rads, pdelta_gt, pmask_gt, num_z, num_x, pix_width, energy, prop_dists, processes, device, save_dir, tag, **kwargs):
    #kwargs collects unused arguments
    func = inspect.getframeinfo(inspect.currentframe()).function

    pdelta_est, _ = ctf(rads, pix_width, energy, prop_dists, rel_regp=10**-8, processes=processes, device=device)
    nrmse, bak_val = nrmse_bak(pdelta_est, pdelta_gt, pmask_gt)
    save_data(os.path.join(save_dir, tag), rads, pdelta_gt, pdelta_est, nrmse, bak_val)

    print("{}: % NRMSE is {}; background {}".format(tag, nrmse, bak_val))
    return nrmse

# params is generated in conftest.py
# Any function that begins with test is a test
def test_ctf_fp(params):
    # Test CTF PR using Fresnel propagation forward model simulation
    nrmse_thresh = 0.2

    func = inspect.getframeinfo(inspect.currentframe()).function
    rads, pdelta_gt, pbeta_gt, pmask_gt = sim_rads(forw_model="FP", **params)
    nrmse = eval_ctf(rads, pdelta_gt, pmask_gt, tag=func, **params)
    assert nrmse < nrmse_thresh

def test_ctf_ctf(params):
    # Test CTF PR using CTF forward model simulation
    nrmse_thresh = 0.2

    func = inspect.getframeinfo(inspect.currentframe()).function
    rads, pdelta_gt, pbeta_gt, pmask_gt = sim_rads(forw_model="CTF", **params)
    nrmse = eval_ctf(rads, pdelta_gt, pmask_gt, tag=func, **params)
    assert nrmse < nrmse_thresh

def eval_tie(rads, pdelta_gt, pmask_gt, num_z, num_x, pix_width, energy, prop_dists, processes, device, save_dir, tag, **kwargs):
    #kwargs collects unused arguments
    pdelta_est, _ = tie(rads[:, :2], pix_width, energy, prop_dists[:2], rel_regp=10**-12, processes=processes, device=device)
    nrmse, bak_val = nrmse_bak(pdelta_est, pdelta_gt, pmask_gt)
    save_data(os.path.join(save_dir, tag), rads, pdelta_gt, pdelta_est, nrmse, bak_val)
    # ERROR: Projection of delta is inverted

    print("{}: % NRMSE is {}".format(tag, nrmse))
    return nrmse

# params is generated in conftest.py
def test_tie_fp(params):
    nrmse_thresh = 0.15

    func = inspect.getframeinfo(inspect.currentframe()).function
    rads, pdelta_gt, pbeta_gt, pmask_gt = sim_rads(forw_model="FP", **params)
    nrmse = eval_tie(rads, pdelta_gt, pmask_gt, tag=func, **params)
    assert nrmse < nrmse_thresh

def test_tie_tie(params):
    nrmse_thresh = 0.15

    func = inspect.getframeinfo(inspect.currentframe()).function
    rads, pdelta_gt, pbeta_gt, pmask_gt = sim_rads(forw_model="TIE", second_order=True, **params)
    
    nrmse = eval_tie(rads, pdelta_gt, pmask_gt, tag=func, **params)
    assert nrmse < nrmse_thresh

def eval_mixed(rads, pdelta_gt, pmask_gt, num_z, num_x, pix_width, energy, prop_dists, processes, device, save_dir, tag, **kwargs):
    #kwargs collects unused arguments

    pdelta_est, _ = mixed(rads, pix_width, energy, prop_dists, rel_regp=10**-5, max_iter=3, processes=processes, device=device)
    nrmse, bak_val = nrmse_bak(pdelta_est, pdelta_gt, pmask_gt)
    save_data(os.path.join(save_dir, tag), rads, pdelta_gt, pdelta_est, nrmse, bak_val)

    print("{}: % NRMSE is {}".format(tag, nrmse))
    return nrmse

# params is generated in conftest.py
def test_mixed(params):
    nrmse_thresh = 1

    func = inspect.getframeinfo(inspect.currentframe()).function
    rads, pdelta_gt, pbeta_gt, pmask_gt = sim_rads(**params)
    nrmse = eval_mixed(rads, pdelta_gt, pmask_gt, tag=func, **params)
    assert nrmse < nrmse_thresh

def eval_nlpr(rads, pdelta_gt, pmask_gt, num_z, num_x, pix_width, energy, prop_dists, processes, device, save_dir, tag, **kwargs):
    #kwargs collects unused arguments
    func = inspect.getframeinfo(inspect.currentframe()).function

    solver_kwargs = {'solver': 'non-native', 'history_size': 64, 'max_iter': 1000, 'line_search_fn': 'Wolfe', 'rec_thresh': 0.5, 'rec_convg': True, 'cost_thresh': 1.0, 'cost_convg': False, 'chk_iters': 5, 'convg_filep': None}   

    pdelta_est, pbeta_est = ctf(rads, pix_width, energy, prop_dists, rel_regp=10**-8, processes=processes, device=device)
    pdelta_est, _ = nlpr(rads, pix_width, energy, prop_dists, pdelta_est, pbeta_est,
                                processes=processes, device=device, solver_kwargs=solver_kwargs) 
    nrmse, bak_val = nrmse_bak(pdelta_est, pdelta_gt, pmask_gt)
    save_data(os.path.join(save_dir, tag), rads, pdelta_gt, pdelta_est, nrmse, bak_val)

    print("{}: % NRMSE is {}".format(tag, nrmse))
    return nrmse

# params is generated in conftest.py
def test_nlpr(params):
    nrmse_thresh = 0.1

    func = inspect.getframeinfo(inspect.currentframe()).function
    rads, pdelta_gt, pbeta_gt, pmask_gt = sim_rads(**params)
    nrmse = eval_nlpr(rads, pdelta_gt, pmask_gt, tag=func, **params)
    assert nrmse < nrmse_thresh

