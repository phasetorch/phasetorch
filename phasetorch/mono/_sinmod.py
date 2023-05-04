import torch
import numpy as np
from phasetorch.util._tutils import get_tensor, get_freqcoord, ptorch_fft, ptorch_ifft
from phasetorch.util._utils import get_wavelength

# Paganin, D., Mayo, S.C., Gureyev, T.E., Miller, P.R. and Wilkins, S.W. (2002), Simultaneous phase and amplitude extraction from a single defocused image of a homogeneous object. Journal of Microscopy, 206: 33-40.
class PaganinTran(torch.nn.Module):
    def __init__(self, device, dtype, N_y, N_x, pix_wid, energy, prop_dist, mag_fact):
        super(PaganinTran, self).__init__()
        pix_wid = pix_wid*mag_fact

        self.device = device
        wlength = get_wavelength(energy)
        self.in_scale = mag_fact**2
        self.filt_scale = np.pi*wlength*prop_dist/mag_fact
        y_coord, x_coord = get_freqcoord(N_y, N_x, pix_wid)
        y_coord = y_coord[np.newaxis, :, :, np.newaxis]
        x_coord = x_coord[np.newaxis, :, :, np.newaxis]
        self.y_coord = get_tensor(y_coord, device=device, dtype=dtype)
        self.x_coord = get_tensor(x_coord, device=device, dtype=dtype)
        self.zero = get_tensor(0.0, device=device, dtype=dtype)

    def forward(self, x, delta_over_beta):
        x = torch.stack((x, torch.zeros_like(x)), dim=-1)
        x = ptorch_fft(self.in_scale*x)

        # prop_dist*delta*4*pi*pi/(mag_fact*mu) # 4*pi*pi is angular->spatial freq
        #   = prop_dist*wlength*delta*4*pi*pi/(mag_fact*4*pi*beta)
        #   = prop_dist*wlength*delta*pi/(mag_fact*beta)
        x = x/(self.filt_scale*delta_over_beta *
               (self.y_coord**2+self.x_coord**2)+1.0)
        x = ptorch_ifft(x)
#        assert torch.isclose(x[:,:,:,1], self.zero).all()
    
        # (2*pi/wlength)*delta/mu = (delta/beta)/2
        return -torch.log(x[:, :, :, 0])*delta_over_beta/2
