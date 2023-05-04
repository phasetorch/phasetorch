import torch
import numpy as np
from skimage.restoration import unwrap_phase
from phasetorch.util._check import assert_dims, assert_Ndists, expand_dims, get_Ndists
from phasetorch.util._proc import MultiProc
from phasetorch.util._pad import calc_padwid, arr_pad, arr_unpad
from phasetorch.util._utils import cproj_cphase, cphase_cproj
from phasetorch.util._tutils import get_tensor, CustomMSELoss
from phasetorch.mono._simtor import cmplxtrans
from phasetorch.mono._nonmod import ConNonLinTran, NonLinTran
from phasetorch.opt._pytorch_lbfgs.functions.LBFGS import FullBatchLBFGS as NNAT_LBFGS
from phasetorch.util._utils import get_wavelength


class NonLPROpt:
    # VS: input *args will be of this form: (Nviews <usualy set to 1>, Npad_y, Npad_x, pix_wid, energy, prop_dists)
    # VS, AM: if we make a function call say self.__init__(*args_full) where args_full is a tuple, then the the first few elements of args_full will be assigned to ...
    #     device, dtype, solver_kwargs, tran_func, verbose, ret_conp, and then the remaining elements in args_full will be assigned to *args
    def __init__(self, device, dtype, solver_kwargs, tran_func, verbose, ret_conp, *args):
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.ret_conp = ret_conp
        # VS, AM: Function handle for regularized loss function: l(y, y_est(x), weights=I) + \beta r(x), where l = data fidelity (likelihood) model and regularization model. Regularization isn't enabled now.
        #     In the below implementation, the loss function takes y, y_est, x and weights as input
        self.loss_fn = CustomMSELoss().compute
        # VS: self.model instantiates an object (say of class NonLinTran) that implements the forward model
        #     The above class inherits from torch.nn.module (torch neural-network). Why so ? I'm guessing NN attributes like self.model().to(device) can be used for GPU accleration and so on.
        #     Consequently self.model() invokes the routine self.model.forward()
        self.model = tran_func(device, dtype, *args)
        self.solver_kwargs = solver_kwargs

        if self.verbose:
            print('LBFGS: {}'.format(solver_kwargs))

        names, self.params = [], []
        for n, v in self.model.named_parameters(recurse=False):
            names.append(n)
            self.params.append(v)

        if self.verbose:
            print('Optimized parameters are {}'.format(names))

    # VS, AM: input init will be of this form: (delta_projs[i:i+1], alpha, gamma) or (delta_projs[i:i+1], beta_projs[i:i+1])
    def run(self, data, weights, *init):
        # VS: init is a list of parameters. For any module/class say myModule that has these parameters as its members, we can instantiate an object say myModule net(), and then use net.parameters() to provide a list of the parameters.
        # Example: [(param.data, param.size()) for param in net.parameters()], or, list(net.parameters())
        init = [get_tensor(v, device=self.device, dtype=self.dtype)
                for v in init]
        data = get_tensor(data, device=self.device, dtype=self.dtype)
        weights = get_tensor(weights, device=self.device, dtype=self.dtype)
        self.model.init(*init)

        if self.solver_kwargs['solver'] == 'native':
            return self.native_lbfgs(data, weights)
        elif self.solver_kwargs['solver'] == 'non-native':
            return self.nonnat_lbfgs(data, weights)
        else:
            print('{} not recognized'.format(self.solver_kwargs['solver']))

    def native_lbfgs(self, data, weights):
        data = torch.sqrt(data) # Square root data is used for phase retrieval

        kwargs = {key: val for key, val in self.solver_kwargs.items()
                  if key != 'solver'}
        optimizer = torch.optim.LBFGS(self.params, **kwargs)
        # print(kwargs, optimizer)
        loss = self.loss_fn(data, self.model(), self.model.rec, weights)
        init_loss = loss.item()

        if self.verbose:
            print('Initial loss: {:.2e}'.format(init_loss))

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            data_pred = self.model()
            lossvl = self.loss_fn(data, data_pred, self.model.rec, weights)
            # print(lossvl)
            if lossvl.requires_grad:
                lossvl.backward()
            return lossvl

        optimizer.step(closure)
        data_sqroot = self.model()
        loss = self.loss_fn(data, data_sqroot, self.model.rec, weights)

        if self.verbose:
            print('Final loss: {:.2e}'.format(loss.item()))

        if self.ret_conp:
            rec = self.model.pr().cpu().numpy()
            convg_params = {
                'final loss': loss.item(), 'initial loss': init_loss, 'any nan': np.any(np.isnan(rec))}
            return rec, convg_params
        else:
            return self.model.pr().cpu().numpy()

    def nonnat_lbfgs(self, data, weights):
        data = torch.sqrt(data) # Square root data is used for phase retrieval

        optimizer = NNAT_LBFGS(self.params, history_size=self.solver_kwargs['history_size'], line_search=self.solver_kwargs['line_search_fn'], dtype=torch.float64 if self.dtype == 'double' else torch.float32, device=self.device)
        loss = self.loss_fn(data, self.model(), self.model.rec, weights)
        init_loss = loss.item()

        if self.verbose:
            print('Initial loss: {:.2e}'.format(init_loss))

        def closure():
            optimizer.zero_grad()
            data_pred = self.model()
            lossvl = self.loss_fn(data, data_pred, self.model.rec, weights)
            return lossvl

        loss_old = closure()
        loss_new = loss_old
        loss_old.backward()

        obj_old = self.model.pr()
        # tol_recon,tol_grad,tol_loss = self.solver_kwargs['tolerance_recon'],self.solver_kwargs['tolerance_grad'],self.solver_kwargs['tolerance_loss']
        itr, maxls, max_maxls, step_maxls = 0, 10, 1000, 10
        total_closure, total_grad = 0, 0

        convg_iters = 0
        nan_present = False
        while itr < self.solver_kwargs['max_iter']:
            options = {'closure': closure, 'current_loss': loss_new,
                       'max_ls': maxls, 'damping': True}
            loss_new, grad_new, _, _, closures_new, grads_new, desc_dir, fail = optimizer.step(
                options=options)
            obj_new = self.model.pr()

            del_rec = 100*torch.mean(torch.abs(obj_new-obj_old)) / \
                torch.abs(torch.mean(obj_old))
            # max_grad = torch.max(grad_new)
            del_loss = (100*torch.abs(loss_new-loss_old)/loss_old).item()

            if torch.any(torch.isnan(obj_new)):
                print('ERROR: Reconstruction failed. NaN detected. Try initializing differently.')
                nan_present = True
                break

            obj_old = torch.clone(obj_new)
            loss_old = torch.clone(loss_new)
            itr = itr + 1
            total_closure = total_closure + closures_new
            total_grad = total_grad + grads_new

            if fail == True and maxls < max_maxls:
                maxls = maxls + step_maxls

            # if del_rec < tol_recon and max_grad < tol_grad and del_loss < tol_loss:
            if ((not self.solver_kwargs['cost_convg']) or del_loss < self.solver_kwargs['cost_thresh']) and ((not self.solver_kwargs['rec_convg']) or del_rec < self.solver_kwargs['rec_thresh']):
                convg_iters = convg_iters + 1
                if convg_iters >= self.solver_kwargs['chk_iters']:
                    break
            else:
                convg_iters = 0

        if itr == self.solver_kwargs['max_iter']:
            print('WARNING: Maximum number of iterations is reached.')

        if self.verbose:
            print('Total iters is {} and max line search iters is {}'.format(itr, maxls))
            print('Total closure evals is {} and total grad evals is {}'.format(
                total_closure, total_grad))
            # print('Recon change is {:.2e}, loss change is {:.2e}, and max grad is {:.2e}'.format(del_rec,del_loss,max_grad))
            print('Perc change in reconstruction is {:.2e}'.format(del_rec))
            print('Perc change in loss is {:.2e}'.format(del_loss))

        data_sqroot = self.model()
        loss = self.loss_fn(data, data_sqroot, self.model.rec, weights)

        if self.verbose:
            print('Final loss: {:.2e}'.format(loss.item()))

        if self.ret_conp:
            rec = obj_old.cpu().numpy()
            nan_present = np.any(np.isnan(rec)) if nan_present is False else nan_present
            convg_params = {'iterations': itr, 'final loss': loss.item(
            ), 'max line-search iterations': maxls, 'initial loss': init_loss, 'any nan': nan_present}
            return rec, convg_params
        else:
            return obj_old.cpu().numpy()


def nlpr(rads, pix_wid, energy, prop_dists, delta_projs, beta_projs, weights=None, processes=1, device='cpu', solver_kwargs=None, mult_FWHM=5, dtype='float', ret_pad=False, verbose=True, ret_conp=False):
    """Non-linear phase retrieval algorithm for imaging of multi-material objects using X-ray phase contrast CT at multiple propagation distances.

    Retrieves the projections of the refractive index decrement and absorption index from normalized radiographs acquired at multiple object to detector propagation distances. This function uses an iterative optimization algorithm that inverts a forward measurement model based on the Fresnel transform. As input, this function takes normalized radiographs. Importantly, this algorithm benefits from an intelligent initialization for the projections of the refractive index decrement (delta), :math:`{\delta}`, and/or absorption index :math:`{\\beta}`. It returns refined estimates for the projections of the refractive index decrement and absorption index. As additional inputs, it also requires specification of the pixel width, X-ray energy, and propagation distances. Other optional inputs that drives the numerical optimization method used for phase retrieval can also be provided. The same length units should be used for all inputs.

    Parameters
    ----------
    rads : numpy.ndarray
        Normalized radiographs.
    pix_wid : float
        Pixel width.
    energy : float
        X-ray energy.
    prop_dists : list of float
        Propagation distances from object to detector (or scintillator).
    delta_projs : numpy.ndarray
        Initial values for the refractive index decrement projections.
    beta_projs : numpy.ndarray
        Initial values for the absorption index projections.
    weights : numpy.ndarray, optional
        Weights are a relative importance measure or relative information measure for the radiograph pixel values. If not provided, then every pixel in `rads` is assigned equal importance.
    processes : int, default=1
        Number of parallel processes. If running on GPUs, this parameter is the total number of available GPUs.
    device : {'cpu', 'cuda'}, default='cpu'
        Target device for running this function. String 'cuda' specifies GPU execution.
    solver_kwargs : dict, optional
        Dictionary of parameters for controlling the LBFGS algorithm used for numerical optimization. Even though optional, it is recommended to intelligently set this parameter. Please see notes for more information.
    mult_FWHM : float, default=5
        Specifies the dominant width of the Fresnel impulse response function in the spatial domain as a multiple of its full width half maximum (FWHM). This is used to determine padding.
    dtype : {'float', 'double'}, default='float'
        Specifies the floating point precision during computation. 'float' uses 32-bit single precision floats and 'double' uses 64-bit double precision floats.
    ret_pad : bool, default=True
        If True, the returned arrays include additional padding.
    verbose : bool, default=False
        If True, print additional messages to aid in debugging.
    ret_conp : bool, default=False
        If True, return parameters related to convergence of optimization algorithm such as final loss function value, etc.

    Returns
    -------
    delta_projs : numpy.ndarray
        Projections of the refractive index decrement.
    beta_projs : numpy.ndarray
        Projections of the absorption index.
    convg_params: dict (Optional)
        Returns convergence parameters if the parameter `ret_conp` is True.

    Notes
    -----
    The shape of `rads` is :math:`(N_{views}, N_{dists}, N_y, N_x)`. If the shape of `rads` is :math:`(N_{views}, N_y, N_x)`, it is equivalent to :math:`N_{dists}=1`. If the shape of `rads` is :math:`(N_y, N_x)`, it is equivalent to :math:`N_{views}=1` and :math:`N_{dists}=1`. The returned arrays `delta_projs` and `beta_projs` are of shape :math:`(N_{views}, N_y, N_x)`.
    It is recommended to set the parameter `solver_kwargs` intelligently. `solver_kwargs` is a python dictionary with several key and value pairs. All keys are python strings (`str`). The value for key `'solver'` is either `'native'` or `'non-native'`. `'native'` refers to the LBFGS solver that is shipped with pytorch. When using this solver, the other solver parameters for LFBGS as defined in `torch.optim.LBFGS <https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html>`_ are specified as key-value pairs. When the value for key `'solver'` is `'non-native'`, this function uses a different `open source implementation <https://github.com/hjmshi/PyTorch-LBFGS>`_ for LBFGS with several key benefits over the `'native'` solver. In this case, the value for key `'history_size'` is an integer (`int`) that specifies the history size for LBFGS. Value for key `'max_iter` is an integer (`int`) that defines the maximum number of LBFGS iterations. Value for key `'line_search_fn'` must be `'Wolfe'`. Value for key `'pct_thresh'` is a floating point threshold (`float`) for percentage change in estimated values. Value for key 'chk_iters' is an integer (`int`) for the number of iterations to check for convergence. LBFGS is stopped when the percentage change in estimates fall below `solver_kwargs['pct_thresh']` for `solver_kwargs['chk_iters']` number of consecutive iterations. Value for key `'cost_convg'` is `False`.

    Examples
    --------
    .. code-block:: python

        if __name__ == '__main__': #Required for parallel compute
            import numpy as np
            import matplotlib.pyplot as plt
            from phasetorch.sim import simdata
            from phasetorch.pr import paganin, nlpr

            #Grid shape 128 x 128. Pixel 0.001mm.
            x, y = np.meshgrid(np.arange(0, 128.0)*0.001,
                               np.arange(0, 128.0)*0.001)
            #Ground-truth projections. Unitary dimension indicates one view.
            projs = np.zeros_like(x)[np.newaxis]
            #Mask for non-zero projections of 0.032mm cylinder.
            mask = (0.032**2 - (x-0.064)**2) > 0
            #Generate path lengths.
            projs[0][mask] = 2*np.sqrt((0.032**2 - (x-0.064)**2)[mask])
            delta_projs, beta_projs = projs*1e-7, projs*1e-8 #delta = 1e-7, beta=1e-8

            #Radiographs at distances of 100mm and 200mm for 20keV X-rays
            rads = simdata(delta_projs, beta_projs, 0.001, 20, [
                           10, 100, 300], processes=1, device='cuda')

            delta_projs_pag = paganin(
                rads[:,1], 0.001, 20, 100, 10, processes=1, device='cuda') #Paganin phase retrieval

            delta_projs_nlpr, beta_projs_nlpr = nlpr(rads, 0.001, 20, [
                                                     10, 100, 300], delta_projs_pag, beta_projs=delta_projs_pag/10, processes=1, device='cuda') #Multi distance NLPR

            #Root mean squared error
            rmse = np.sqrt(np.mean((delta_projs_pag - delta_projs)**2))
            print('Percentage RMSE with Paganin is {}'.format(
                100*rmse/np.mean(delta_projs)))
            #Root mean squared error
            rmse = np.sqrt(np.mean((delta_projs_nlpr - delta_projs)**2))
            print('Percentage RMSE for delta with multi-distance NLPR is {}'.format(100* \
                  rmse/np.mean(delta_projs)))

            #Estimated projections of delta using multi-distance NLPR
            plt.imshow(delta_projs_nlpr[0])
            plt.show()
    """

    if len(prop_dists) == 1:
        print('WARNING: Multiple propagation distances may be necessary in the absence of material constraints.')
    elif len(prop_dists) == 0:
        raise ValueError('prop_dists is empty')

    return nlprbase(rads, pix_wid, energy, prop_dists, delta_projs, beta_projs, alpha=None, gamma=None, weights=weights, processes=processes, device=device, solver_kwargs=solver_kwargs, mult_FWHM=mult_FWHM, dtype=dtype, ret_pad=ret_pad, verbose=verbose, ret_conp=ret_conp)

def tropt_nlprcon(delta, beta, num_xy, pix_wid, energy, low_trans=0.01):
    """
    For constrained NLPR, compute the constraint parameters of ``alpha`` and ``gamma`` such that the transmission values are approximately scaled between 0 and ``low_trans``.

    Parameters
    ----------
    delta : float
        Refractive index decrement.
    beta : float
        Absorption index.
    num_xy : int
        Maximum number of pixels along a ray path in the xy-plane of object (plane perpendicular to axis of rotation).
    pix_wid : float
        Pixel width.
    low_trans : float
        Lower bound for the real-valued transmission.

    Returns
    -------
    alpha : float
        Constraint parameter for the real part of the transmission.
    gamma : float
        Constraint parameter for the imaginary part of the transmission.
    """

    wnumber = (2*np.pi/get_wavelength(energy))
    alpha = -beta*pix_wid*num_xy*wnumber/np.log(low_trans)
    gamma = -delta*pix_wid*num_xy*wnumber/np.log(low_trans)

    return alpha, gamma


def nlprcon(rads, pix_wid, energy, prop_dists, delta_projs, alpha, gamma, weights=None, processes=1, device='cpu', solver_kwargs=None, mult_FWHM=5, dtype='float', ret_pad=False, verbose=True, ret_conp=False):
    """Non-linear phase retrieval algorithm for imaging of homogeneous or weakly absorbing objects using X-ray phase contrast CT at a single propagation distance.

    Retrieves the projections of the refractive index decrement and absorption index from normalized radiographs acquired at a single object to detector propagation distance. This function uses an iterative optimization algorithm that inverts a forward measurement model based on the Fresnel transform. As input, this function takes normalized radiographs. Importantly, this algorithm benefits from an intelligent initialization for the projections of the refractive index decrement (delta), :math:`{\delta}`. It returns refined estimates for the projections of the refractive index decrement (delta) and absorption index (beta). As additional inputs, it also requires specification of the pixel width, X-ray energy, and propagation distances. Other optional inputs that drives the numerical optimization method used for phase retrieval can also be provided. The same length units should be used for all inputs.

    Parameters
    ----------
    rads : numpy.ndarray
        Normalized radiographs.
    pix_wid : float
        Pixel width.
    energy : float
        X-ray energy.
    prop_dists : list of float
        Propagation distances from object to detector (or scintillator).
    delta_projs : numpy.ndarray
        Initial values for the refractive index decrement projections.
    alpha : float
        Real value of the estimated transmission function's exponent. Recommended setting is `alpha=1` and `gamma=delta/beta`.
    gamma : float
        Imaginary value of the estimated transmission function's exponent. Recommended setting is `alpha=1` and `gamma=delta/beta`.
    weights : numpy.ndarray, optional
        Weights are a relative importance measure or relative information measure for the radiograph pixel values. If not provided, then every pixel in `rads` is assigned equal importance.
    processes : int, default=1
        Number of parallel processes. If running on GPUs, this parameter is the total number of available GPUs.
    device : {'cpu', 'cuda'}, default='cpu'
        Target device for running this function. String 'cuda' specifies GPU execution.
    solver_kwargs : dict, optional
        Dictionary of parameters for controlling the LBFGS algorithm used for numerical optimization. Even though optional, it is recommended to intelligently set this parameter. Please see notes for more information.
    mult_FWHM : float, default=5
        Specifies the dominant width of the Fresnel impulse response function in the spatial domain as a multiple of its full width half maximum (FWHM). This is used to determine padding.
    dtype : {'float', 'double'}, default='float'
        Specifies the floating point precision during computation. 'float' uses 32-bit single precision floats and 'double' uses 64-bit double precision floats.
    ret_pad : bool, default=True
        If True, the returned arrays include additional padding.
    verbose : bool, default=False
        If True, print additional messages to aid in debugging.
    ret_conp : bool, default=False
        If True, return parameters related to convergence of optimization algorithm such as final loss function value, etc.

    Returns
    -------
    delta_projs : numpy.ndarray
        Projections of the refractive index decrement.
    beta_projs : numpy.ndarray
        Projections of the absorption index. Redundant since `beta_projs` can be uniquely determined from `delta_projs`, `alpha`, and `gamma`.
    convg_params: dict (Optional)
        Returns convergence parameters if the parameter `ret_conp` is True.

    Notes
    -----
    The shape of `rads` is :math:`(N_{views}, N_{dists}, N_y, N_x)`. If the shape of `rads` is :math:`(N_{views}, N_y, N_x)`, it is equivalent to :math:`N_{dists}=1`. If the shape of `rads` is :math:`(N_y, N_x)`, it is equivalent to :math:`N_{views}=1` and :math:`N_{dists}=1`. The returned arrays `delta_projs` and `beta_projs` are of shape :math:`(N_{views}, N_y, N_x)`.
    It is recommended to set the parameter `solver_kwargs` intelligently. `solver_kwargs` is a python dictionary with several key and value pairs. All keys are python strings (`str`). The value for key `'solver'` is either `'native'` or `'non-native'`. `'native'` refers to the LBFGS solver that is shipped with pytorch. When using this solver, the other solver parameters for LFBGS as defined in `torch.optim.LBFGS <https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html>`_ are specified as key-value pairs. When the value for key `'solver'` is `'non-native'`, this function uses a different `open source implementation <https://github.com/hjmshi/PyTorch-LBFGS>`_ for LBFGS with several key benefits over the `'native'` solver. In this case, the value for key `'history_size'` is an integer (`int`) that specifies the history size for LBFGS. Value for key `'max_iter` is an integer (`int`) that defines the maximum number of LBFGS iterations. Value for key `'line_search_fn'` must be `'Wolfe'`. Value for key `'pct_thresh'` is a floating point threshold (`float`) for percentage change in estimated values. Value for key 'chk_iters' is an integer (`int`) for the number of iterations to check for convergence. LBFGS is stopped when the percentage change in estimates fall below `solver_kwargs['pct_thresh']` for `solver_kwargs['chk_iters']` number of consecutive iterations. Value for key `'cost_convg'` is `False`.

    Examples
    --------
    .. code-block:: python

        if __name__ == '__main__': #Required for parallel compute
            import numpy as np
            import matplotlib.pyplot as plt
            from phasetorch.sim import simdata
            from phasetorch.pr import paganin, nlprcon

            #Grid shape 128 x 128. Pixel 0.001mm.
            x, y = np.meshgrid(np.arange(0, 128.0)*0.001,
                               np.arange(0, 128.0)*0.001)
            #Ground-truth projections. Unitary dimension indicates one view.
            projs = np.zeros_like(x)[np.newaxis]
            #Mask for non-zero projections of 0.032mm cylinder.
            mask = (0.032**2 - (x-0.064)**2) > 0
            #Generate path lengths.
            projs[0][mask] = 2*np.sqrt((0.032**2 - (x-0.064)**2)[mask])
            delta_projs, beta_projs = projs*1e-7, projs*1e-8 #delta = 1e-7, beta=1e-8

            #Radiographs at distances of 100mm and 200mm for 20keV X-rays
            rads = simdata(delta_projs, beta_projs, 0.001, 20,
                           100, processes=1, device='cuda')
            rads = rads.squeeze(1) #Remove unitary dimension

            #Paganin phase retrieval
            delta_projs_pag = paganin(
                rads, 0.001, 20, 100, 10, processes=1, device='cuda')
            delta_projs_nlpr, beta_projs_nlpr = nlprcon(
                rads, 0.001, 20, 100, delta_projs_pag, alpha=1, gamma=10, processes=1, device='cuda') #Single distance NLPR

            #Root mean squared error
            rmse = np.sqrt(np.mean((delta_projs_pag - delta_projs)**2))
            print('Percentage RMSE with Paganin is {}'.format(
                100*rmse/np.mean(delta_projs)))
            #Root mean squared error
            rmse = np.sqrt(np.mean((delta_projs_nlpr - delta_projs)**2))
            print('Percentage RMSE with single-distance NLPR is {}'.format(100* \
                  rmse/np.mean(delta_projs)))

            #Estimated projections of delta using single-distance NLPR
            plt.imshow(delta_projs_nlpr[0])
            plt.show()
    """
    return nlprbase(rads, pix_wid, energy, prop_dists, delta_projs, beta_projs=None, alpha=alpha, gamma=gamma, weights=weights, processes=processes, device=device, solver_kwargs=solver_kwargs, mult_FWHM=mult_FWHM, dtype=dtype, ret_pad=ret_pad, verbose=verbose, ret_conp=ret_conp)


def nlprbase(rads, pix_wid, energy, prop_dists, delta_projs, beta_projs=None, alpha=None, gamma=None, weights=None, processes=1, device='cpu', solver_kwargs=None, mult_FWHM=5, dtype='float', ret_pad=False, verbose=True, ret_conp=False):
    if verbose:
        print('processes={}, device={}, dtype={}'.format(
            processes, device, dtype))
        print('pix_wid={}, energy={}, prop_dists={}'.format(
            pix_wid, energy, prop_dists))

    pix_wid = float(pix_wid)
    energy = float(energy)
    if solver_kwargs is None:
        solver_kwargs = {'solver':'non-native', 'history_size':64, 'max_iter':1e4, 'line_search_fn':'Wolfe', 'rec_thresh':0.5, 'rec_convg':True, 'cost_thresh':1.0, 'cost_convg':True, 'chk_iters':5}

    N_dists, prop_dists = get_Ndists(prop_dists)
    assert_dims(rads, dmin=2, dmax=4)
    # VS: data dimensions - (#views, #dists, #detector-rows, #detector-columns)
    # Add view and distance dimensions if they don't exist
    rads = expand_dims(rads, dmax=2, dapp=0)
    rads = expand_dims(rads, dmax=3, dapp=1)
    # Check if distance dimension of rads is equal to N_dists
    assert_Ndists(rads, N_dists, 1)
    N_y, N_x = rads.shape[-2], rads.shape[-1]

    if weights is not None:
        assert_dims(weights, dmin=2, dmax=4)
        weights = expand_dims(weights, dmax=2, dapp=0)
        weights = expand_dims(weights, dmax=3, dapp=1)
        assert_Ndists(weights, N_dists, 1)
        # VS: data dimensions - (#views, #dists, #detector-rows, #detector-columns)
    else:
        weights = np.ones_like(rads)
        weights = weights/(N_dists*N_y*N_x)

    # if not np.isclose(np.sum(weights), 1.0):
    #    weights = weights/np.sum(weights) #Normalize
    #    print('Normalizing weights')

    # VS: Refer to the paper: Voelz and Rogemann paper 2009, "Digital simulation of scalar optical diffraction: revisiting chrip function sampling criteria"
    #     The chirp function (used in the Fresnel convolution kernel) is not band-limited and this can cause aliasing.
    #     Our sampling frequency Fs is given by (1/pix_wid).
    #     So we window (band-limit) the chirp function so that the maximum frequency B in the chirp satisfies the Nquist sampling criertion.
    pad_y, pad_x = calc_padwid(
        N_y, N_x, pix_wid, max(prop_dists), energy, mult_FWHM)
    Npad_y = N_y + pad_y[0] + pad_y[1]
    Npad_x = N_x + pad_x[0] + pad_x[1]

    rads = arr_pad(rads, pad_y, pad_x, mode='edge')
    weights = arr_pad(weights, pad_y, pad_x, mode='constant', const=0.0)
    assert rads.shape == weights.shape

    assert rads.shape[0] == delta_projs.shape[0], 'First dimension of rads must equal first dimension of phase'
    if alpha is not None or gamma is not None:
        assert (alpha is not None) and (gamma is not None)
        alpha, gamma = float(alpha), float(gamma)
        # VS: pretty neat that python allows the name of the class to be treated as a variable. See my comments in _proc.py, launch_proc() for more details.
        tran_func = ConNonLinTran
        delta_projs = arr_pad(delta_projs, pad_y, pad_x, mode='edge')
        delta_projs = cproj_cphase(delta_projs, energy)
        if beta_projs is not None:
            print('WARNING: beta_projs is not None and will be ignored')
    elif beta_projs is not None:
        # VS: pretty neat that python allows the name of the class to be treated as a variable. See my comments in _proc.py, launch_proc() for more details.
        tran_func = NonLinTran
        delta_projs = cproj_cphase(
            arr_pad(delta_projs, pad_y, pad_x, mode='edge'), energy)
        beta_projs = cproj_cphase(
            arr_pad(beta_projs, pad_y, pad_x, mode='edge'), energy)
        # cproj_cphase(.) is used to multiply by 2*pi/wavelength
        assert rads.shape[0] == beta_projs.shape[0], 'First dimension of rads must equal first dimension of absorp'
    else:
        raise ValueError(
            'ERROR: Either beta_projs or beta_over_delta must be supplied and not None')

    # VS: the 1 in the below arguments refers to the number of views.
    #     It is actually the number of views to processed by each device/process.
    args = (device, dtype, solver_kwargs, tran_func, verbose, ret_conp,
            1, Npad_y, Npad_x, pix_wid, energy, prop_dists)
    launcher = MultiProc(NonLPROpt, args, processes, device)
    # VS: rads is a 4-D array of size (NumViews = #processes, Ndistances, Ny, Nx)
    for i in range(rads.shape[0]):
        if alpha is not None and gamma is not None:
            args = (rads[i:i+1], weights[i:i+1],
                    delta_projs[i:i+1], alpha, gamma)
        else:
            args = (rads[i:i+1], weights[i:i+1],
                    delta_projs[i:i+1], beta_projs[i:i+1])
        launcher.queue(args)
    rvals = launcher.launch_onebyone()

    if ret_conp:
        projs_np, convg_params = [], []
        for r in rvals:
            assert len(r) == 2
            projs_np.append(r[0])
            convg_params.append(r[1])
        projs_np = np.concatenate(projs_np, axis=0)
    else:
        projs_np = np.concatenate(rvals, axis=0)

    delta_projs = projs_np[..., 0]
    beta_projs = projs_np[..., 1]
    del projs_np  # clear data
    if alpha is None or gamma is None:
        # VS: Importantly this step performs phase-unwrapping, allowing us to use FBP after phase-retrieval to compute voxel-wise refractive-index decrement
        for i in range(delta_projs.shape[0]):
            delta_projs[i] = unwrap_phase(delta_projs[i], seed=0)

    # VS: The following commands are useful
    # 1) ar.real and ar.imag where ar is a complex array of any dimension
    # 2) arr_unpad(ar_padded, pad_y, pad_x) where ar is a 2D array returns unpadded array. This is a function from "_pad.py"
    # 3) img = Image.fromarray(ar) for a real-valued 2D array allows you to then save the image as a tif with ...
    #    img.save('<filename>.tif')

    # VS, AM: This is merely scaling images with the inverse of 2*pi/lambda

    delta_projs = cphase_cproj(delta_projs, energy)
    beta_projs = cphase_cproj(beta_projs, energy)

    opt_ret = {}
    if ret_pad:
        opt_ret['mask'] = arr_unpad(delta_projs, pad_y, pad_x, ret_mask=True)
    else:
        delta_projs = arr_unpad(delta_projs, pad_y, pad_x)
        beta_projs = arr_unpad(beta_projs, pad_y, pad_x)

    if ret_conp:
        opt_ret['convg_params'] = convg_params

    if ret_pad or ret_conp:
        return delta_projs, beta_projs, opt_ret
    else:
        return delta_projs, beta_projs
