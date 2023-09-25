import pytest
from test_mono.utils import sim_rads, save_data, read_yaml_params
from phasetorch.mono.pr import paganin, nlprcon
from phasetorch.mono.pr import tropt_nlprcon
from skimage.metrics import normalized_root_mse
import matplotlib.pyplot as plt
import inspect
import os

def eval_paganin(rads, pdelta_gt, num_z, num_x, pix_width, energy, prop_dists, processes, device, delta_gt, beta_gt, save_dir, tag, **kwargs):
    #kwargs collects unused arguments

    pdelta_est = paganin(rads, pix_width, energy, prop_dists, 
                            float(sum(delta_gt))/sum(beta_gt), processes=processes, device=device) 
    
    nrmse = normalized_root_mse(pdelta_est, pdelta_gt, normalization="euclidean")    
    save_data(os.path.join(save_dir, tag), rads, pdelta_gt, pdelta_est, nrmse)

    print("{}: % NRMSE is {}".format(tag, nrmse))
    return nrmse

# params is generated in conftest.py
def test_paganin(params):
    nrmse_thresh = 0.15

    func = inspect.getframeinfo(inspect.currentframe()).function
    rads, pdelta_gt, pbeta_gt, _ = sim_rads(**params)
    nrmse = eval_paganin(rads, pdelta_gt, tag=func, **params)
    assert nrmse < nrmse_thresh

def eval_nlprcon(rads, pdelta_gt, num_z, num_x, pix_width, energy, prop_dists, processes, device, delta_gt, beta_gt, save_dir, tag, constraint="alpha1", **kwargs):
    #kwargs collects unused arguments
    func = inspect.getframeinfo(inspect.currentframe()).function

    delta_gt = float(sum(delta_gt))/len(delta_gt)
    beta_gt = float(sum(beta_gt))/len(beta_gt)
    
    if constraint == "alpha1": 
        alpha, gamma = 1.0, delta_gt/beta_gt
    elif constraint == "gamma1":
        alpha, gamma = beta_gt/delta_gt, 1.0 
    elif constraint == "tropt":
        alpha, gamma = tropt_nlprcon(delta_gt, beta_gt, num_x, pix_width, energy)
    else:
        raise ValueError("Constraint not recognized.")

    solver_kwargs = {'solver': 'non-native', 'history_size': 64, 'max_iter': 1000, 'line_search_fn': 'Wolfe', 'rec_thresh': 0.5, 'rec_convg': True, 'cost_thresh': 1.0, 'cost_convg': False, 'chk_iters': 5, 'convg_filep': None}   

    pdelta_est = paganin(rads, pix_width, energy, prop_dists, 
                                delta_gt/beta_gt, processes=processes, device=device)
    pdelta_est, _ = nlprcon(rads, pix_width, energy, prop_dists, pdelta_est, alpha, gamma,
                                processes=processes, device=device, solver_kwargs=solver_kwargs) 
    nrmse = normalized_root_mse(pdelta_est, pdelta_gt, normalization="euclidean") 
    save_data(os.path.join(save_dir, tag), rads, pdelta_gt, pdelta_est, nrmse)

    print("{}: % NRMSE is {}".format(tag, nrmse))
    return nrmse

# params is generated in conftest.py
def test_nlprcon_alpha1(params):
    nrmse_thresh = 0.1

    func = inspect.getframeinfo(inspect.currentframe()).function
    rads, pdelta_gt, pbeta_gt, _ = sim_rads(**params)
    nrmse = eval_nlprcon(rads, pdelta_gt, tag=func, constraint="alpha1", **params)
    assert nrmse < nrmse_thresh

def test_nlprcon_tropt(params):
    nrmse_thresh = 0.1

    func = inspect.getframeinfo(inspect.currentframe()).function
    rads, pdelta_gt, pbeta_gt, _ = sim_rads(**params)
    nrmse = eval_nlprcon(rads, pdelta_gt, tag=func, constraint="tropt", **params)
    assert nrmse < nrmse_thresh
  
