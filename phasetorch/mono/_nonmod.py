import torch
import numpy as np
from phasetorch.mono._fresmod import FresnelProp
from phasetorch.util._tutils import get_tensor

class ConNonLinTran(torch.nn.Module):
    def __init__(self, device, dtype, N_views, N_y, N_x, pix_wid, energy, prop_dist):
        super(ConNonLinTran, self).__init__()
        assert dtype == 'double' or dtype == 'float'
        self.model = FresnelProp(device, dtype, N_y, N_x, pix_wid, energy, prop_dist)
       
        np_dtype = np.float64 if dtype=='double' else np.float32 
        rec = np.ones((N_views,1,N_y,N_x), dtype=np_dtype, order='C')

        self.rec = get_tensor(rec, device=device, dtype=dtype, grad=True) #projections of delta

    def pr(self):
        assert self.rec.size(1)==1
        rec = torch.abs(self.rec[:,0])
        p = -torch.log(rec)
        beta_projs = p*self.alpha
        delta_projs = p*self.gamma
        
        return torch.stack((delta_projs, beta_projs), dim=-1).detach()
        #For returned tensor, [:,:,0] is delta_projs and [:,:,1] is beta_projs

    def init(self, delta_projs, alpha, gamma):
        self.alpha, self.gamma = alpha, gamma
        assert self.rec.size(1)==1
        self.rec.data[:,0:1] = torch.exp(-delta_projs/gamma)

        #alpha_np, gamma_np = alpha.cpu().numpy(), gamma.cpu().numpy()
#        print('-----------------Init-------------------------')
#        print('alpha = {}, gamma = {}'.format(alpha_np, gamma_np))
#        print('exp(-alpha) = {}, exp(-gamma) = {}'.format(np.exp(-alpha_np), np.exp(-gamma_np)))
#        print('Avg of delta_projs is {}'.format(torch.mean(delta_projs)))
#        print('Avg of self.rec is {}'.format(torch.mean(self.rec)))
#        print('Min of delta_projs is {}'.format(torch.min(delta_projs)))
#        print('Min of self.rec is {}'.format(torch.min(self.rec)))
#        print('Max of delta_projs is {}'.format(torch.max(delta_projs)))
#        print('Max of self.rec is {}'.format(torch.max(self.rec)))

    def forward(self):
        rec = torch.abs(self.rec)
        phase = torch.log(rec)*self.gamma
        x = rec**self.alpha
        rec_real = x*torch.cos(phase)
        rec_imag = x*torch.sin(phase)
        
        x = torch.stack((rec_real, rec_imag), dim=-1)
        return self.model(x)

class NonLinTran(torch.nn.Module):
    def __init__(self, device, dtype, N_views, N_y, N_x, pix_wid, energy, prop_dists):
        super(NonLinTran, self).__init__()
        assert dtype == 'double' or dtype == 'float'

        # VS: self.model instantiates an object of class FresnelProp that implements the forward model
        #     The above class inherits from torch.nn.module (torch neural-network). Why so ? I'm guessing NN attributes like self.model().to(device) can be used fro GPU accleration and so on.
        #     Consequently self.model() invokes the routine self.model.forward()
        self.model = FresnelProp(device, dtype, N_y, N_x, pix_wid, energy, prop_dists)
        # VS, AM: Why is the reconstruction size set to (Nviews, 1, Ny, Nx, 2) ? My guess the radiographs are (Nviews, Ndist, Ny, Nx, 1).
        #     So we can conveniently use broadcasting on the reconstruction while defining the loss function across the Ndist different distances.
        
        np_dtype = np.float64 if dtype=='double' else np.float32 
        rec = np.ones((N_views,1,N_y,N_x,2), dtype=np_dtype, order='C') #Last dim is for delta/beta
        
        self.rec = get_tensor(rec, device=device, dtype=dtype, grad=True) #VS: this is a tensor, more specifically a parameter

    def pr(self):
        assert self.rec.size(1) == 1

        beta_projs = torch.sqrt(self.rec[...,0]**2 + self.rec[...,1]**2)
        beta_projs = -torch.log(beta_projs)            
        delta_projs = -torch.atan2(self.rec[...,1], self.rec[...,0])
        projs = torch.stack((delta_projs, beta_projs), dim=-1)

        return projs[:,0].detach() #projs[...,0] is delta_projs and projs[...,1] is beta_projs

    def init(self, delta_projs, beta_projs):
         # VS: re-intializes projections from 0 tensor --> to any provided value. note that .data can be used for any tensor, not just parameter sub-class of tensors.
        assert self.rec.size(1)==1
        x = torch.exp(-beta_projs) #Compute absorption
        self.rec.data[:,0:1,:,:,0:1] = (x*torch.cos(-delta_projs))[...,np.newaxis]
        self.rec.data[:,0:1,:,:,1:2] = (x*torch.sin(-delta_projs))[...,np.newaxis]

    # VS: The forward() function has to be defined in this way since NonLinTran inherits from torch.nn.module.
    def forward(self):
        x = self.rec
        return self.model(x)
