import torch
import numpy as np
from phasetorch.util._utils import get_wavelength
from torch.nn.functional import conv2d

def ptorch_fft(x, norm="backward"):
    assert x.size(-1) == 2
    x = torch.view_as_complex(x)
    x = torch.fft.fft2(x, norm=norm)
    x = torch.view_as_real(x)
    #x = torch.fft(x, signal_ndim=2, normalized=True)
    return x 

def ptorch_ifft(x, norm="backward"):
    assert x.size(-1) == 2
    x = torch.view_as_complex(x)
    x = torch.fft.ifft2(x, norm=norm)
    x = torch.view_as_real(x)
    #x = torch.ifft(x, signal_ndim=2, normalized=True)
    return x

def get_tensor(arr, device, dtype, grad=False):
    if arr is None:
        return None

    assert dtype == 'double' or dtype == 'float'
    dtype = torch.float64 if dtype == 'double' else torch.float32
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr).to(dtype=dtype, device=device)
    else:
        arr = torch.tensor([arr], requires_grad=grad).to(dtype=dtype, device=device)
    return torch.nn.Parameter(arr, requires_grad=grad)

# VS: This is used for defining the convolutional kernel h for fresnel diffraction
#     For the convolution kernel, grid order is changed from -Nx/2-->0-->Nx/2 to 0-->Nx/2,-Nx/2-->0 (-ve axis wrapped around as per convention/requirement of FFT / IFFT routine)
#     If the above kernel, is real-valued and symmetric (about -Nx/2-->0-->Nx/2), its fourier representation will be real-valued.
#     Same grid-wrapping is NOT required of the input signal that is convolved with th3 abive kernel.
def get_spacecoord(N_y, N_x, pix_wid):
    y_coord = np.arange(0.0, float(N_y), 1.0)
    y_coord[y_coord >= N_y//2] = N_y-y_coord[y_coord >= N_y//2]
    x_coord = np.arange(0.0, float(N_x), 1.0)
    x_coord[x_coord >= N_x//2] = N_x-x_coord[x_coord >= N_x//2]

    y_coord, x_coord = y_coord*pix_wid, x_coord*pix_wid
    y_coord, x_coord = np.meshgrid(y_coord, x_coord, indexing='ij')
    return y_coord, x_coord

# VS: Order is changed from -Nx/2-->0-->Nx/2 to 0-->Nx/2,-Nx/2-->0 because of fftshifting in the FFT
def get_freqcoord(N_y, N_x, pix_wid):
    Fy_wid = 1.0/(N_y*pix_wid)
    Fx_wid = 1.0/(N_x*pix_wid)

    y_coord = np.arange(0.0, float(N_y), 1.0)
    y_coord[y_coord >= N_y//2] = N_y-y_coord[y_coord >= N_y//2]
    x_coord = np.arange(0.0, float(N_x), 1.0)
    x_coord[x_coord >= N_x//2] = N_x-x_coord[x_coord >= N_x//2]

    y_coord, x_coord = y_coord*Fy_wid, x_coord*Fx_wid
    y_coord, x_coord = np.meshgrid(y_coord, x_coord, indexing='ij')
    return y_coord, x_coord

def get_xderiv(x):
    #sz = x.size(3)
    #grad = torch.zeros_like(x)
    #grad[:,:,:,1:sz-2] = x[:,:,:,2:sz-1]-x[:,:,:,0:sz-3]
    #grad[:,:,:,0] = x[:,:,:,1] - x[:,:,:,sz-1] #Rotational convolution
    #grad[:,:,:,sz-1] = x[:,:,:,0] - x[:,:,:,sz-2]
    #return grad/2
    sobel = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]]).to(x.get_device()).type(x.type())
    sobel = sobel.unsqueeze(0).unsqueeze(0)/8.0
    grad = conv2d(x, sobel, padding="same") 
    assert grad.size() == x.size()
    return grad

def get_yderiv(x):
    #sz = x.size(2)
    #grad = torch.zeros_like(x)
    #grad[:,:,1:sz-2] = x[:,:,2:sz-1]-x[:,:,0:sz-3]
    #grad[:,:,0] = x[:,:,1] - x[:,:,sz-1] #Rotational convolution
    #grad[:,:,sz-1] = x[:,:,0] - x[:,:,sz-2]
    #return grad/2
    sobel = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]]).to(x.get_device()).type(x.type())
    sobel = sobel.unsqueeze(0).unsqueeze(0)/8.0
    grad = conv2d(x, sobel, padding="same") 
    assert grad.size() == x.size()
    return grad

class CustomMSELoss:
    def __init__(self):
        super(CustomMSELoss, self).__init__()
        self.loss = torch.nn.MSELoss(reduction='none')

    def compute(self, y, y_est, x, weights=1):
        loss = torch.sum(weights*((y-y_est)**2))

        # loss = torch.sum(weights*(torch.abs(torch.sqrt(y)-y_est)**2))
        # VS: Here ... in tensor indexing means as many : as needed
        #     Similar to torch.fft, dct_2d is computed over batches of 2-D arrays.
        #     For eg dct_2d(x) where x is an array of size n2*n1*n0, means n2 individual dcts will be computed, each with size n1*n0.
        # VS: The Re/Im parts have physical significance - the former represents the net attenuation (material absorption, like CT radiograph) and the latter reprents the net phase-change
        #if self.reg_dct > 0:
        #    import torch_dct as dct
        #    dct_real = dct.dct_2d(x[...,0], norm='ortho')
        #    dct_imag = dct.dct_2d(x[...,1], norm='ortho')
        #    dct_all = torch.stack((dct_real,dct_imag), dim=0)
        #    # L1 norm pernalty on the dct image
        #    dct_loss = self.reg_dct*torch.mean(torch.abs(dct_all))
        #else:
        #    dct_loss = torch.FloatTensor([0.0]).to('cuda')

        #loss = forw_loss + dct_loss

        return loss
