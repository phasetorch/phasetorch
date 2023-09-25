from phasetorch.util._proc import MultiProc
from phasetorch.inv.nn import INRFit
from phasetorch.mono._fresmod import FresnelProp
import torch

import numpy as np
from skimage.restoration import unwrap_phase
from phasetorch.util._check import assert_dims, assert_Ndists, expand_dims, get_Ndists
from phasetorch.util._proc import MultiProc
from phasetorch.util._pad import calc_padwid, arr_pad, arr_unpad
from phasetorch.util._utils import cproj_cphase, cphase_cproj
from phasetorch.mono._mulmod import CTFTran, TIETran

# wrapper class for unconstrained inversion using INR of projection space parameters
class UnconProjINR(torch.nn.Module):
    def __init__(self, device, dtype, forw_model, N_y, N_x, pix_wid, energy, prop_dists):
        super().__init__()

        self.device, self.dtype = device, dtype
        if forw_model == "Fresnel":
            self.tran_model = FresnelProp(device, dtype, N_y, N_x, pix_wid, energy, prop_dists)
            # Fresnel transform returns square root magnitude. 
        elif forw_model == "TIE":
            self.tran_model = TIETran(device, dtype, N_y, N_x, pix_wid, energy, prop_dists, None, 1e-5)
            # Unused dummy parameter setting rel_reg_par=1e-5.
        elif forw_model == "CTF":
            self.tran_model = CTFTran(device, dtype, N_y, N_x, pix_wid, energy, prop_dists, None, 1e-5)
            # Unused dummy parameter setting rel_reg_par=1e-5.
        else:
            raise ValueError("For forw_model, choose between Fresnel, TIE, or CTF!")
   
    # return stack of delta/beta projections
    # ensures common return between this class and ConProjINR 
    def pr(self, delta_projs, beta_projs):
        projs = torch.stack((delta_projs, beta_projs), dim=-1)
        return projs   

    # INR produces real and imag projections
    def forward(self, delta_projs, beta_projs):
        # delta_projs -> ref idx dec projections
        # beta_projs -> abs idx projecions
        # projs dim: views x y-axis x x-axis
        return self.tran_model.prop(delta_projs, beta_projs)

# wrapper class for constrained optimization of projection space parameters
# proportionality constraint between projections of delta and beta
class ConProjINR(torch.nn.Module): 
    # all parameters from forw_model must be passed as a dictionary during multiprocessing
    def __init__(self, device, dtype, forw_model, N_y, N_x, pix_wid, energy, prop_dists, delta_over_beta):
        super().__init__()

        self.device, self.dtype = device, dtype
        self.delta_over_beta = delta_over_beta        

        if forw_model == "Fresnel":
            self.tran_model = FresnelProp(device, dtype, N_y, N_x, pix_wid, energy, prop_dists)
            # Fresnel transform returns square root magnitude. 
        elif forw_model == "TIE":
            self.tran_model = TIETran(device, dtype, N_y, N_x, pix_wid, energy, prop_dists, None, 1e-5)
            # Unused dummy parameter setting rel_reg_par=1e-5.
        elif forw_model == "CTF":
            self.tran_model = CTFTran(device, dtype, N_y, N_x, pix_wid, energy, prop_dists, None, 1e-5)
            # Unused dummy parameter setting rel_reg_par=1e-5.
        else:
            raise ValueError("For forw_model, choose between Fresnel, TIE, or CTF!")
        
    def pr(self, delta_projs):
        return delta_projs.unsqueeze(dim=-1) # add 1-dim to denote only delta projs

    def forward(self, delta_projs):
        # delta_projs -> ref idx dec projections
        # beta_projs -> abs idx projecions
        beta_projs = delta_projs/self.delta_over_beta
        return self.tran_model.prop(delta_projs, beta_projs)

def inr_pr(rads, pix_wid, energy, prop_dists, delta_projs=None, beta_projs=None, delta_over_beta=None, weights=None, init_projs=False, delta_scale=None, beta_scale=None, inr_sigma=1.0, forw_model="Fresnel", processes=1, device="cpu", solver_kwargs=None, mult_FWHM=1, win_size=None, dtype="float", ret_pad=False, ret_conp=False):
    pix_wid = float(pix_wid)
    energy = float(energy)

    N_dists, prop_dists = get_Ndists(prop_dists)
    assert_dims(rads, dmin=2, dmax=4)

    # add view and distance dimensions if they don't exist
    # if no of dims is 2, add one extra dimension to indicate one view
    rads = expand_dims(rads, dmax=2, dapp=0)
    # if no of dims is 3, add one extra dimension for one distance
    rads = expand_dims(rads, dmax=3, dapp=1)
    # dim for rads: views x distances x y-axis x x-axis 
    
    # check if distance dimension of rads is equal to N_dists
    assert_Ndists(rads, N_dists, 1)
    N_y, N_x = rads.shape[-2], rads.shape[-1]

    # INR training uses Adam optimizer. Hence, the solver parameters are for Adam.
    if solver_kwargs is None:
        solver_kwargs = {"lr": 1e-3, "init_batch": N_x//2, "init_epochs": 100 if init_projs else 0, 
                "fit_batch": N_x//4, "fit_epochs": 100, "thresh": 0.1, "lr_gamma": 0.95}
        # lr is learning rate
        # max_epochs is maximum number of epochs
        # each epoch iterates over all possible pixel locations for the top-left corner of a patch
        # thresh is for determining convergence 

    if weights is not None:
        assert_dims(weights, dmin=2, dmax=4)
        weights = expand_dims(weights, dmax=2, dapp=0)
        weights = expand_dims(weights, dmax=3, dapp=1)
    else:
        weights = np.ones_like(rads)
        # normalize weights over distances, and pixels of y-axis/x-axis.
        weights = weights/(N_dists*N_y*N_x)

    # D. G. Voelz and M. C. Roggemann, Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria and consequences,b[1]D. G. Voelz and M. C. Roggemann, bbppl. Opt., vol. 48, no. 32, p. 6132, Nov. 2009, doi: 10.1364/AO.48.006132.
    # padding determined by the drop-off of discrete Fresnel transform in space domain.
    # padding is largest for the highest distance.
    pad_y, pad_x = calc_padwid(
        N_y, N_x, pix_wid, max(prop_dists), energy, mult_FWHM)

    Npad_y = N_y + pad_y[0] + pad_y[1]
    Npad_x = N_x + pad_x[0] + pad_x[1]

    rads = arr_pad(rads, pad_y, pad_x, mode="edge")
    # 0 padding for weights for ignoring loss in the padded region
    weights = arr_pad(weights, pad_y, pad_x, mode="constant", const=0.0)
    assert rads.shape == weights.shape
  
    if not np.issctype(delta_projs): # check if not a scalar 
        # verify same number of view dimesnion for rads and delta_projs 
        assert rads.shape[0] == delta_projs.shape[0], "First dimension of rads must equal first dimension of phase."
    else:
        # delta_projs is only used for scaling of INR outputs if it is a scalar
        delta_projs = np.ones_like(rads[:, 0])*delta_projs
        assert init_projs == False, "For initialization, array of delta_projs must be provided."
 
    assert forw_model == "Fresnel" or forw_model == "TIE" or forw_model == "CTF" # supported forw models 
    # compute window and average pad size 
    if forw_model == "TIE":
        # support of 5x5 for gradient followed by divergence in forw model.
        win_size = 7
        pad_size = win_size//2
    else:
        pad_size = round((pad_y[0] + pad_y[1] + pad_x[0] + pad_x[1])/4)
        win_size = 2*pad_size + 1
        win_size = 2 if win_size < 2 else win_size # minimum window of size 2

    print("Window = {}, padding = {}, N_y = {}, N_x = {}".format(win_size, pad_size, N_y, N_x), flush=True)

    if delta_over_beta is not None:
        delta_over_beta = float(delta_over_beta)
        # for constraint, only delta_projs is the free variable for optimization.
        # beta_projs and delta_projs are related by a 1-1 relation.
        
        tran_func = ConProjINR
        # all parameters from forw_model are passed in a dictionary
        tran_pars = {"forw_model": forw_model, "N_y": win_size + 2*pad_size, "N_x": win_size + 2*pad_size, 
                        "pix_wid": pix_wid, "energy": energy, 
                        "prop_dists": prop_dists, "delta_over_beta": delta_over_beta}

        delta_projs = arr_pad(delta_projs, pad_y, pad_x, mode="edge")
        
        # preferably must be the average value of phase projections
        inr_scales = [np.mean(delta_projs) if delta_scale is None else delta_scale] 
        inr_nout = len(inr_scales) 

        if beta_projs is not None:
            print("WARNING: beta_projs is not None and will be ignored", flush=True)

    elif beta_projs is not None:
        # without constraints, both delta_projs and beta_projs are free variables.
        tran_func = UnconProjINR
        # padding on all four sides before running through forward model
        tran_pars = {"forw_model": forw_model, "N_y": win_size + 2*pad_size, "N_x": win_size + 2*pad_size, "pix_wid": pix_wid,
                        "energy": energy, "prop_dists": prop_dists}

        delta_projs = arr_pad(delta_projs, pad_y, pad_x, mode='edge')
        beta_projs = arr_pad(beta_projs, pad_y, pad_x, mode='edge')

        # scaling of INR outputs is the average value of real/imag transmission/projection
        inr_scales = [np.mean(delta_projs) if delta_scale is None else delta_scale, 
                        np.mean(beta_projs) if beta_scale is None else beta_scale]
        inr_nout = len(inr_scales)
 
        assert rads.shape[0] == beta_projs.shape[0], "First dimension of rads must equal first dimension of absorp"
    else:
        raise ValueError(
            "ERROR: Either beta_projs or beta_over_delta must be supplied and not None")
        
    inr_nchan = 16*round(np.sqrt(N_x))
    inr_params = {"sigma": inr_sigma, "scales": inr_scales, "num_chan": inr_nchan, 
                    "num_out": inr_nout, "pre_layers": 3, "post_layers": 2, "batch_norm": False}

    print("Scaling for INR outputs are {}".format(inr_scales), flush=True)
    print("INR parameters {}".format(inr_params), flush=True)
    print("Transmission parameters {}".format(tran_pars), flush=True)
    print("Solver parameters {}".format(solver_kwargs), flush=True)       
 
    args = (device, dtype, tran_func, tran_pars, inr_params, win_size, pad_size, solver_kwargs, ret_conp)
    launcher = MultiProc(INRFit, args, processes, device)
    
    for i in range(rads.shape[0]):
        # use dim non-preserving indexing to remove view dim
        # view dim in forw model is used for batches of patches
        if delta_over_beta is not None:
            # newaxis adds an extra last dim to represent one basis
            args = (rads[i], weights[i], delta_projs[i][..., np.newaxis])
        else:
            args = (rads[i], weights[i], delta_projs[i][..., np.newaxis], beta_projs[i][..., np.newaxis])

        launcher.queue(args)
    rvals = launcher.launch_onebyone()

    # collect estimated projection and loss data from all views
    if ret_conp:
        projs_np, losses = [], []
        for r in rvals:
            assert len(r) == 2
            projs_np.append(r[0])
            losses.append(r[1])
        #projs_np = np.concatenate(projs_np, axis=0)
        projs_np = np.stack(projs_np, axis=0) # no view dim
    else:
        #projs_np = np.concatenate(rvals, axis=0)
        projs_np = np.stack(rvals, axis=0) # no view dim

    delta_projs = projs_np[..., 0]
    if delta_over_beta is None:
        beta_projs = projs_np[..., 1]
    else:
        assert projs_np.shape[-1] == 1
        beta_projs = delta_projs/delta_over_beta

    del projs_np  # clear data
            
    # convert to phase for phase unwrapping
    # cproj_cphase(.) is used to multiply by 2*pi/wavelength
    delta_projs = cproj_cphase(delta_projs, energy)

    if delta_over_beta is None: # meaning absence of constraints
        # phase unwrapping is necessary in the absence of material constraints
        for i in range(delta_projs.shape[0]):
            delta_projs[i] = unwrap_phase(delta_projs[i], seed=0)

    # divides by the wavenumber
    delta_projs = cphase_cproj(delta_projs, energy)
    # beta_projs = cphase_cproj(beta_projs, energy)

    # either remove padding here or use mask to remove padding later
    # retaining the padding may be useful for downstream tasks
    if ret_pad:
        opt_ret['mask'] = arr_unpad(delta_projs, pad_y, pad_x, ret_mask=True)
    else:
        delta_projs = arr_unpad(delta_projs, pad_y, pad_x)
        beta_projs = arr_unpad(beta_projs, pad_y, pad_x)

    # return losses only if convergence parameters are desired
    if ret_conp:
        opt_ret['losses'] = losses

    if ret_pad or ret_conp:
        return delta_projs, beta_projs, opt_ret
    else:
        return delta_projs, beta_projs
