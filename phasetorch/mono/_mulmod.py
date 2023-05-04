from phasetorch.util._tutils import get_tensor, get_freqcoord, get_xderiv, get_yderiv, ptorch_fft, ptorch_ifft
from phasetorch.util._utils import get_wavelength
import torch
import numpy as np

# Langer, Max, et al. "Quantitative comparison of direct phase retrieval algorithms in inb# Paganin, D., Mayo, S.C., Gureyev, T.E., Miller, P.R. and Wilkins, S.W. (2002), Simultaneous phase and amplitude extraction from a single defocused image of a homogeneous object. Journal of Microscopy, 206: 33-40.
# Zabler, S., et al. "Optimization of phase contrast imaging using hard x rays." Review of Scientific Instruments 76.7 (2005): 073705.
# Paganin, David, and Keith A. Nugent. "Noninterferometric phase imaging with partially coherent light." Physical review letters 80.12 (1998): 2586.
# Guigay, Jean Pierre, et al. "Mixed transfer function and transport of intensity approach for phase retrieval in the Fresnel region." Optics letters 32.12 (2007): 1617-1619.

class CTFTran(torch.nn.Module):
    def __init__(self, device, dtype, N_y, N_x, pix_wid, energy, prop_dists, reg_par, rel_reg_par):
        super(CTFTran, self).__init__()
        self.device = device
        prop_dists = np.array(prop_dists, dtype=np.double, order='C')
        prop_dists = prop_dists[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
        wlength = get_wavelength(energy)
        y_coord, x_coord = get_freqcoord(N_y, N_x, pix_wid)
        y_coord = y_coord[np.newaxis,np.newaxis,:,:,np.newaxis]
        x_coord = x_coord[np.newaxis,np.newaxis,:,:,np.newaxis]

        cosv = np.cos(np.pi*wlength*prop_dists*(x_coord**2+y_coord**2))
        sinv = np.sin(np.pi*wlength*prop_dists*(x_coord**2+y_coord**2))
        self.cosv = get_tensor(cosv, device=device, dtype=dtype)
        self.sinv = get_tensor(sinv, device=device, dtype=dtype)

        self.A = get_tensor(np.sum(sinv*cosv, axis=1),
                            device=device, dtype=dtype)
        self.B = get_tensor(np.sum(sinv*sinv, axis=1),
                            device=device, dtype=dtype)
        self.C = get_tensor(np.sum(cosv*cosv, axis=1),
                            device=device, dtype=dtype)
        self.zero = get_tensor(0.0, device=device, dtype=dtype)

        Delta = self.B*self.C - self.A*self.A
        assert not (reg_par is not None and rel_reg_par is not None)
        if reg_par is not None:
            self.out_scale = 1.0/(2*Delta+reg_par)
            print("min/mean/max/std 2*Delta", torch.min(2*Delta), torch.mean(2*Delta), torch.max(2*Delta), torch.std(2*Delta), "reg_par", reg_par)
        elif rel_reg_par is not None:
            reg_par = rel_reg_par*torch.max(2*Delta)
            self.out_scale = 1.0/(2*Delta+reg_par)
            print("reg_par is {} when rel_reg_par is {}".format(reg_par, rel_reg_par))
        else:
            self.out_scale = 1.0/(2*Delta)

        # VS: added for self.prop()
        self.wlength, self.sizes_padded, self.dtype = wlength, (N_y, N_x), dtype

    def forward(self, x):
        x = torch.stack((x, torch.zeros_like(x)), dim=-1)
        x = ptorch_fft(x)
        Isin = torch.sum(x*self.sinv, dim=1)
        Icos = torch.sum(x*self.cosv, dim=1)

        phase = self.out_scale*(self.C*Isin - self.A*Icos)
        phase = ptorch_ifft(phase)
        assert torch.isclose(phase[:, :, :, 1], self.zero, atol=1e-3).all()
        phase = -phase[:, :, :, 0]
        # Negative sign due to paper's defn

        absorp = self.out_scale*(self.A*Isin - self.B*Icos)
        absorp = ptorch_ifft(absorp)
        assert torch.isclose(absorp[:, :, :, 1], self.zero, atol=1e-3).all()
        absorp = absorp[:, :, :, 0]
        return phase.detach(), absorp.detach()

    # VS: added this for mono-energetic CTF propagator.
    #     <beta/delta>_projs is a 3d numpy array of shape (#views, N_y, N_x) that represents the projection line-intergal of beta/delta
    def prop(self, beta_projs, delta_projs):
        assert (beta_projs.ndim==3) and (delta_projs.ndim==3) and (beta_projs.shape == delta_projs.shape)
        assert (beta_projs.shape[1:] == self.sizes_padded)
        Npad_y, Npad_x = self.sizes_padded

        # scale projections by 2*pi/lambda and shift to device
        wnum  = 2*np.pi/self.wlength
        projs = np.stack((wnum*beta_projs, -wnum*delta_projs), axis=0)
        projs = get_tensor(projs, device=self.device, dtype=self.dtype) #size: (#bd, #views, Npad_y, Npad_x) where #bd=2 denotes beta/delta

        # dirac delta in fourier domain (account for scaling constant)
        ones = torch.stack((torch.ones((Npad_y, Npad_x), device=self.device), torch.zeros((Npad_y, Npad_x), device=self.device)), dim=-1) #size: (Npad_y, Npad_x, 2)
        dirac_delta = ptorch_fft(ones)                                                        #size: (Npad_y, Npad_x, 2)

        # fourier domain forward operation
        x    = torch.stack((projs, torch.zeros_like(projs)), dim=-1) #size: (#bd, #views, Npad_y, Npad_x, 2)
        x_f  = ptorch_fft(x)                                         #size: (#bd, #views, Npad_y, Npad_x, 2)
        x_f  = x_f.permute(1,2,3,0,4)                                #size: (#views, Npad_y, Npad_x, #bd, 2)
        # reshape and stack the cos/sin chirps from original shape of (1,#dists, Npad_y, Npad_x, 1)
        chirps = torch.stack( (-2*self.cosv[0, ..., 0], 2*self.sinv[0, ..., 0]), dim=-1) #size: (#dists, Npad_y, Npad_x, #bd) where #bd=2
        chirps = chirps.permute(1,2,0,3)  #size: (Npad_y, Npad_x, #dists, #bd)
        y_f = torch.matmul(chirps, x_f)   #size: (#views, Npad_y, Npad_x, #dists, 2)
        y_f = y_f.permute(0,3,1,2,4)      #size: (#views, #dists, Npad_y, Npad_x, 2)
        y_f = y_f + dirac_delta           #size: (#views, #dists, Npad_y, Npad_x, 2)
        # revert to spatial domain. Unpad will be done externally
        y = ptorch_ifft(y_f)              #size: (#views, #dists, Npad_y, Npad_x, 2)
        return y[...,0] # In X-ray intensity space. Retains negative values.


class TIETran(torch.nn.Module):
    def __init__(self, device, dtype, N_y, N_x, pix_wid, energy, prop_dists, reg_par, rel_reg_par):
        super(TIETran, self).__init__()
        self.device = device
        self.pix_wid = pix_wid
        prop_dists = np.array(prop_dists, dtype=np.double, order='C')

        wlength = get_wavelength(energy)
        y_coord, x_coord = get_freqcoord(N_y, N_x, pix_wid)
        y_coord = y_coord[np.newaxis,:,:,np.newaxis]
        x_coord = x_coord[np.newaxis,:,:,np.newaxis]
        xy_coord = x_coord**2+y_coord**2

        assert not (reg_par is not None and rel_reg_par is not None)
        if rel_reg_par is not None:
            reg_par = rel_reg_par*np.max(xy_coord)
            print("reg_par is {} when rel_reg_par is {}".format(reg_par, rel_reg_par))
        elif reg_par is None:
            reg_par = 0.0

        invlap = -(1.0/(4*np.pi*np.pi))*(1.0/(xy_coord+reg_par))
        self.invlap = get_tensor(invlap, device=device, dtype=dtype)
        self.out_scale = -(2*np.pi/wlength)
        self.zero = get_tensor(0.0, device=device, dtype=dtype)

        self.deriv_x = torch.nn.Conv2d(1, 1, kernel_size=3, bias=False, padding="same", padding_mode="replicate")
        kernel = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])/8.0
        self.deriv_x.weight.data = get_tensor(kernel, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

        self.deriv_y = torch.nn.Conv2d(1, 1, kernel_size=3, bias=False, padding="same", padding_mode="replicate")
        kernel = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])/8.0
        self.deriv_y.weight.data = get_tensor(kernel, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

        # VS: added for prop()
        self.wnum, self.dists, self.dtype  = (2*np.pi/wlength), prop_dists, dtype

    def forward(self, z):
        self.diff_dists = self.dists[1]-self.dists[0]

        assert len(
            self.dists) == 2, 'Only two propagation distances are supported'

        assert z.size(
            1) == 2, 'Second dimension must be at 2 propagation distances'

        delz = (z[:, 1:2]-z[:, 0:1])/self.diff_dists
        delz = torch.stack((delz, torch.zeros_like(delz)), dim=-1)
        delz = ptorch_fft(delz)
        delz = ptorch_ifft(delz*self.invlap)
        assert torch.isclose(delz[:, :, :, :, 1], self.zero).all()
        delz = delz[:, :, :, :, 0]

        #grad_x, grad_y = get_xderiv(delz), get_yderiv(delz)
        grad_x, grad_y = self.deriv_x(delz), self.deriv_y(delz)
        if self.diff_dists > 0:
            grad_x, grad_y = grad_x/z[:, 0:1], grad_y/z[:, 0:1]
        else:
            grad_x, grad_y = grad_x/z[:, 1:2], grad_y/z[:, 1:2]

        #delz = get_xderiv(grad_x) + get_yderiv(grad_y)
        delz = self.deriv_x(grad_x) + self.deriv_y(grad_y)

        # Divide by pixel width for grad calc
        delz = delz/(self.pix_wid*self.pix_wid)
        delz = torch.stack((delz, torch.zeros_like(delz)), dim=-1)
        delz = ptorch_fft(delz)
        delz = ptorch_ifft(delz*self.invlap)
        assert torch.isclose(delz[:, :, :, :, 1], self.zero).all()
        assert delz.size(1) == 1

        # Negative sign due to paper's defn
        ph = -delz[:, 0, :, :, 0]*self.out_scale
        if self.diff_dists > 0:
            ab = -torch.log(z[:, 0])
        else:
            ab = -torch.log(z[:, 1])

        return ph.detach(), ab.detach()

    ##### VS: TO DO: include scaling constant of 1/self.pix_wid within self.deriv_{x/y}() so that it can be ignored (removed) in the wrappers below
    #         This would let us modularly plug-in any 1-D derivative operator including gaussian derivatives ...
    #         and the below wrappers that accordingly define gradient and divergence would still hold.

    # VS: wrappers on top of discrete 1-D gradient computation methods self.deriv_{x/y}
    def gradient_op(self, x):
        grad = torch.stack((self.deriv_x(x), self.deriv_y(x)), dim=-1)/self.pix_wid
        return grad

    def divergence_op(self, x):
        assert (x.size()[-1]==2)
        div = (self.deriv_x(x[...,0]) + self.deriv_y(x[...,1]))/self.pix_wid
        return div

    # VS: Below routine needed for more precise version of TIE model that accounts for 2nd order terms
    # VS: Analytically compute Fourier^-1{  \int_x I_0(x) * |f \cdot \nabla v(x)|^2 exp{-j 2\pi f \cdot x} dx } ...
    #     where x, f denote continuous 2D spatial position and frequency respectively, and I_0() and v() are scalar functions in x
    def second_order_invFourier_op(self, I_0, v):
        # scaling constant for derivative operator (here self.deriv_{x/y})
        pix_wid2 = self.pix_wid * self.pix_wid
        # ceofficienst of f_1^2, f_2^2, f_1f_2  that we get from expanding I_0(x) * |f \cdot \nabla v(x)|^2
        v_grad = self.gradient_op(v)
        h1 =  I_0 * v_grad[...,0] * v_grad[...,0]
        h2 =  I_0 * v_grad[...,1] * v_grad[...,1]
        h3 =  I_0 * v_grad[...,0] * v_grad[...,1] * 2
        # terms we get from inverse fourier transform of const*{f_1^2 F[h_1(x)] + f_2^2 F[h_2] + f_1f_2 F[h_3]} where F denotes fourier-transform and const=-4*pi*pi
        h1 = self.deriv_x(self.deriv_x(h1))/pix_wid2
        h2 = self.deriv_y(self.deriv_y(h2))/pix_wid2
        h3 = self.deriv_x(self.deriv_y(h3))/pix_wid2
        # sum individual terms and account for scaling constant -4*pi*pi
        h  = (h1 + h2 + h3)/(-4*np.pi*np.pi)
        return h


    # VS: added mono-energetic TIE forward propagator
    #     Importantly, we return intensity rather than amplitude = sqrt{intensity}
    #     <beta/delta>_projs is a 3d numpy array of shape (#views, N_y, N_x) that represents the projection line-intergal of beta/delta
    def prop(self, beta_projs, delta_projs, second_order=False):
        assert (beta_projs.ndim==3) and (delta_projs.ndim==3) and (beta_projs.shape == delta_projs.shape)
        dists = get_tensor(self.dists, device=self.device, dtype=self.dtype)               #size: (#dists)
        dists = dists.unsqueeze(dim=0)                                                     #size: (1, #dists)
        B    =  self.wnum * get_tensor(beta_projs,  device=self.device, dtype=self.dtype)  #size: (#views, Ny, Nx)
        phi  = -self.wnum * get_tensor(delta_projs, device=self.device, dtype=self.dtype)  #size: (#views, Ny, Nx)
        I_0  = torch.exp(-2*B)                                                             #size: (#views, Ny, Nx)
        diff = self.divergence_op((I_0.unsqueeze(dim=-1) * self.gradient_op(phi)))         #size: (#views, Ny, Nx)
        diff = (-1/self.wnum) * torch.matmul(diff.unsqueeze(dim=-1), dists)                #size: (#views, Ny, Nx, #dists)
        y    = I_0.unsqueeze(dim=-1) + diff                                                #size: (#views, Ny, Nx, #dists)
        if(second_order):
            print('Using second order terms')
            diff_second_order = self.second_order_invFourier_op(I_0, B) + self.second_order_invFourier_op(I_0, phi)  #size: (#views, Ny, Nx)
            diff_second_order = torch.matmul(diff_second_order.unsqueeze(dim=-1), (dists*dists))                     #size: (#views, Ny, Nx, #dists)
            diff_second_order = (-(np.pi*np.pi)/(self.wnum*self.wnum)) * diff_second_order                           #size: (#views, Ny, Nx, #dists)
            y = y + diff_second_order                                                                                #size: (#views, Ny, Nx, #dists)
        y = y.permute(0,3,1,2)                                                             #size: (#views, #dists, Ny, Nx)
        #y = torch.nn.functional.relu(y)
        return y


class MixedTran(torch.nn.Module):
    def __init__(self, device, dtype, N_y, N_x, pix_wid, energy, prop_dists, reg_par, rel_reg_par, tol, max_iter):
        super(MixedTran, self).__init__()
        self.device = device
        self.tol = tol
        self.max_iter = max_iter
        self.pix_wid = pix_wid

        # VS: Assumes prop_dista may not be sorted, finds minimum dist, and then forms a new version of the same where prop_dists does not include minimum dist.
        prop_dists = np.array(prop_dists, dtype=np.double, order='C')
        self.zidx = np.nonzero(prop_dists == np.min(prop_dists))
        assert len(self.zidx) == 1
        assert len(self.zidx[0]) == 1
        self.zidx = self.zidx[0][0]
        prop_dists = (prop_dists[:self.zidx], prop_dists[self.zidx+1:])
        prop_dists = np.concatenate(prop_dists, axis=0)
        prop_dists = prop_dists[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]

        wlength = get_wavelength(energy)
        y_coord, x_coord = get_freqcoord(N_y, N_x, pix_wid)
        y_coord = y_coord[np.newaxis, np.newaxis, :, :, np.newaxis]
        x_coord = x_coord[np.newaxis, np.newaxis, :, :, np.newaxis]

        cosv = np.cos(np.pi*wlength*prop_dists*(x_coord**2+y_coord**2))
        cosv = cosv*(wlength*prop_dists)/(2*np.pi)
        AD = 2*np.sin(np.pi*wlength*prop_dists*(x_coord**2+y_coord**2))
        ADsq = np.sum(AD*AD, axis=1, keepdims=True)

        assert not (reg_par is not None and rel_reg_par is not None)
        if rel_reg_par is not None:
            reg_par = rel_reg_par*np.max(ADsq)
            print("reg_par is {} when rel_reg_par is {}".format(reg_par, rel_reg_par))
        elif reg_par is None:
            reg_par = 0.0

        self.cosv = get_tensor(cosv, device=device, dtype=dtype)
        self.AD = get_tensor(AD, device=device, dtype=dtype)
        self.AD2reg = get_tensor(ADsq+reg_par, device=device, dtype=dtype)
        self.zero = get_tensor(0.0, device=device, dtype=dtype)

        self.deriv_x = torch.nn.Conv2d(1, 1, kernel_size=3, bias=False, padding="same", padding_mode="replicate")
        kernel = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])/8.0
        self.deriv_x.weight.data = get_tensor(kernel, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

        self.deriv_y = torch.nn.Conv2d(1, 1, kernel_size=3, bias=False, padding="same", padding_mode="replicate")
        kernel = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])/8.0
        self.deriv_y.weight.data = get_tensor(kernel, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

    def forward(self, z):
        assert z.dim() == 4
        # VS: The below is just initializing phase \psi to be 0, but why do we need fourier transforms for the same ?
        psi_f = torch.zeros_like(z[:, 0:1])
        psi_f = torch.stack((psi_f, torch.zeros_like(psi_f)), dim=-1)
        psi = ptorch_ifft(psi_f)
        assert torch.isclose(psi[:, :, :, :, 1], self.zero).all()
        psi = psi[:, :, :, :, 0]

        z0 = z[:, self.zidx:self.zidx+1]
        z0_f = ptorch_fft(torch.stack((z0, torch.zeros_like(z0)), dim=-1))
        assert z0.size(1) == 1
        zall = torch.cat((z[:, :self.zidx], z[:, self.zidx+1:]), dim=1)
        zall_f = ptorch_fft(torch.stack((zall, torch.zeros_like(zall)), axis=-1))

        #lnz0 = torch.log(z0)
        #g0_x, g0_y = get_xderiv(z0), get_yderiv(z0)
        g0_x, g0_y = self.deriv_x(z0), self.deriv_y(z0)

        psi_perc, iters = self.tol, 0
        while psi_perc >= self.tol and iters < self.max_iter:
            #DeltaD = get_xderiv(psi*g0_x) + get_yderiv(psi*g0_y)
            DeltaD = self.deriv_x(psi*g0_x) + self.deriv_y(psi*g0_y)
            DeltaD = DeltaD/(self.pix_wid**2)

            DeltaD = torch.stack((DeltaD, torch.zeros_like(DeltaD)), dim=-1)
            DeltaD = self.cosv*ptorch_fft(DeltaD)

            psi_f_new = self.AD*(zall_f-z0_f-DeltaD)
            psi_f_new = torch.sum(psi_f_new, dim=1, keepdim=True)/self.AD2reg
            psi_new = ptorch_ifft(psi_f_new)
            psi_new = psi_new[:, :, :, :, 0]

            psi_diff = torch.mean(torch.abs(psi_new-psi))
            psi_perc = 100*psi_diff/torch.mean(torch.abs(psi_new))
            iters = iters + 1

            psi_f = torch.clone(psi_f_new)
            psi = torch.clone(psi_new)

        print('Percentage change is {} and total iterations is {}'.format(
            psi_perc, iters))
        assert psi.size(1) == 1
        psi = -psi[:, 0:1]/z0
        # Negative sign due to paper's defn
        ab = -torch.log(z0)
        return psi[:, 0].detach(), ab[:, 0].detach()
