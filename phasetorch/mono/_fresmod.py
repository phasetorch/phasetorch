import torch
import numpy as np
import sys
from phasetorch.util._tutils import get_tensor, get_freqcoord, get_spacecoord, ptorch_fft, ptorch_ifft
from phasetorch.util._utils import get_wavelength

### VS: This is a Fresnel propagator used for forward-modeling ###
### and non-linear phase-retrieval in the case of Mono-energetic X-ray source. ###

# VS: Since FresnelProp inherits from torch.nn, FresnelProp() is equivalent to FresnelProp.forward()
# The processing is batched over multiple propagation distances
# The linear operator FresnelProp.transform is represented in the fourier domain by a tensor of size (1, #dists, #rows, #columns, 2)

class FresnelProp(torch.nn.Module):
    def __init__(self, device, dtype, N_y, N_x, pix_wid, energy, prop_dists):
        super(FresnelProp, self).__init__()

        wlength = get_wavelength(energy)
        assert dtype == 'double' or dtype == 'float'
        np_dtype = np.float64 if dtype == 'double' else np.float32
        prop_dists = prop_dists.astype(np_dtype, order='C')

        fres_trans = []
        # Transfer function for each distance
        # Ref: David G. Voelz and Michael C. Roggemann, "Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria and consequences," Appl. Opt. 48, 6132-6142 (2009)
        for i in range(len(prop_dists)):
            dist = prop_dists[i]
            pix_liml = wlength*dist/(pix_wid*min(N_y, N_x))
            pix_limh = wlength*dist/(pix_wid*max(N_y, N_x))
            if pix_wid >= pix_liml:
                # VS: Express the Fresnel convolution integral in the Fourier domain 
                # (i.e. an analytical Fourier domain reprsentation for H(u,v)).
                # This is what we use by default.
                # print('Fresnel transform in Fourier domain for {}'.format(dist))
                y_coord, x_coord = get_freqcoord(N_y, N_x, pix_wid)
                y_coord = y_coord[np.newaxis]
                x_coord = x_coord[np.newaxis]
                phase_mults = -np.pi*wlength*dist
                Fxy_sq = phase_mults*(y_coord**2+x_coord**2)
                tran = np.exp(1j*Fxy_sq)
                # size: (1, Ny, Nx, 2)
                tran = np.stack((np.real(tran), np.imag(tran)), axis=-1)
                tran = get_tensor(tran, device=device, dtype=dtype)
            elif pix_wid <= pix_limh:
                raise ValueError(
                    'Forward model is not verified when pixel size < '.format(pix_limh))
                # VS: This is the convlutional kernel function that is precomputed and its FFT taken so that it can be applied in the fourier domain.
                # The Fresnel interfence pattern f_D(x,y) of a given optical field f(x,y) can be computed as ...
                # f_D(x',y') = c \int_x \int_y f(x,y) h(x'-x, y'-y) dx dy, where h(x,y) = exp{+j pi/(\lambda*z) [x^2+y^2] } and c is a constant that is only a function of the detector-plane distance z.
                # So in fourier domain F_D(u,v) = F(u,v)H(u,v).
                # Below lines precompute H.
                # print('Fresnel transform in space domain for {}'.format(dist))
                lz = wlength*dist
                # VS: This is meshgrid. (y_coord[i,j], x_coord[i,j]) gives the 2-D pixel position in spatial units at coordinates i,j.
                y_coord, x_coord = get_spacecoord(N_y, N_x, pix_wid)
                # VS, AM: Note the np.newaxis is needed for broadcasting over the view dimension.
                y_coord = y_coord[np.newaxis]
                x_coord = x_coord[np.newaxis]
                xy_sq = (np.pi/lz)*(y_coord**2+x_coord**2)
                # VS: quadratic phase term on detector plane
                tran = (1.0/lz)*np.exp(1j*xy_sq)
                # VS: Use stack to generate complex number from 2 separate components.
                tran = np.stack((np.real(tran), np.imag(tran)), axis=-1)
                tran = get_tensor(tran, device=device, dtype=dtype)
                # VS: 2-D fft complex-->complex. See comments in forward() for more details about ptorch_fft().
                #     The last dimension must be of size 2 representing Re/Im parts.
                tran = ptorch_fft(tran)
            else:
                raise ValueError(
                    'Optimal transform not found. Please adjust radiograph dimensions.')
            fres_trans.append(tran)

        # VS: fresnel transform for multiple distances.
        #     tensor size: (1, #dists, Ny, Nx, 2)
        self.transform = torch.stack(fres_trans, dim=1)

    # VS: The forward() function has to be defined in this way since FresnelProp inherits from torch.nn.module.
    #     The input x is 5D tensor with shape (#views, #dists=1, #rows, #columns, 2) representing transmission plane (complex-valued) optical field.
    #     The fresnel tranform T in the fourier domain is a 5D tensor with shape (#views=1, #dists, #rows, #columns, 2)
    #     We use broadcasting to do complex-multiplication of (T, x) and then retain only magnitude of the product.
    #     The output is a 4D tensor of size (#views, #dists, #rows, #columns).
    def forward(self, x):
        # VS: Complex arithmetic here. ptorch_fft() is complex --> complex transform. So last dimension must be of size 2 representing Re/Im parts
        assert x.size(-1) == 2, 'Last dimension of x must be 2'
        # VS: If x is an array of N-dimensions, then batches of 2-D ffts are computed.
        #     For example ptorch_fft(x) where an array of size n2*n1*n0*2, means n2 individual complex-->complex 2-D ffts will be computed, each with size n1*n0.
        x = ptorch_fft(x)
        # VS: This is just element-wise complex multiplication between x and self.transform, ...
        #     where the latter is Fourier domain reprsentation of Fresnel convolutional kernel
        y_real = x[:, :, :, :, 0]*self.transform[:, :, :, :, 0] - \
            x[:, :, :, :, 1]*self.transform[:, :, :, :, 1]
        y_imag = x[:, :, :, :, 0]*self.transform[:, :, :, :, 1] + \
            x[:, :, :, :, 1]*self.transform[:, :, :, :, 0]
        # VS: Just use stack operation to form a complex number a+jb from 2 separate components a,b. So output size after stacking will be (size(a),2)
        x = torch.stack((y_real, y_imag), dim=-1)
        # VS: complex-->complex IFFT, last dimension must be of size 2 representing Re/Im parts.
        x = ptorch_ifft(x)
        x = x[:, :, :, :, 0]*x[:, :, :, :, 0] + x[:, :, :, :, 1]*x[:, :, :, :, 1]
        return torch.sqrt(x)

