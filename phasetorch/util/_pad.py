import numpy as np
from phasetorch.util._utils import get_wavelength

def next_fft_len(N,factors=[2,3,5]):
    while True:
        temp = N
        for f in factors:
            while temp % f == 0:
                temp /= f
        if temp == 1:
            return N
        N = N + 1
    raise ValueError('Fast length for FFT computation error. N = {}'.format(N))

# VS: The fresnel convolution filter has a long length, so the input optical field (image) will be need to be padded to avoid artficats from circular convolutions.
# VS: We calculate padding required to input image in the case where the chirp convolution filter is sampled in the fourier domain.
#     Note that prop_dist is a scalar (only the maximum distance is used)
def calc_padwid(N_y,N_x,pix_width,prop_dist,energy,mult_FWHM=1,verbose_flag=False):
    wlength = get_wavelength(energy)
    # Effective Filter width (length): Length of convoltion filter where entries are non-zero
    # if optimal fourier-domain sampling (NO over-sampling, Nquist criterion is exactly met) is performed on the fourier domain representation, then ...
    # the spatial domain representation obtained from fourier-inversion will not have any extra zeros padded, ...
    # else there will padded zeros are not essential for the convolution operation.
    fwidth = mult_FWHM*(wlength*prop_dist/pix_width) # units - mm
    fwidth = int(np.ceil(fwidth/pix_width))          # units - pixels
    # VS: If mult_FWHM=1, and radix-based size requirement is dropped ...
    #     you can show that the overall padding in each dimension does NOT exceed N_y and N_x.
    Npad_y = next_fft_len(N_y+fwidth)
    Npad_x = next_fft_len(N_x+fwidth)
    assert Npad_y>0 and Npad_x>0,'Padded lengths must be positive'

    pad_y,pad_x = (Npad_y-N_y),(Npad_x-N_x)
    rgt_y,rgt_x = pad_y//2,pad_x//2
    lft_y,lft_x = pad_y-rgt_y,pad_x-rgt_x

    if(verbose_flag):
        print('fwidth is {} pixels'.format(fwidth))
        print('N_y, N_x = {}, {}'.format(N_y, N_x))
        print('Npad_y, Npad_x = {}, {}'.format(Npad_y, Npad_x))

    return ((lft_y,rgt_y),(lft_x,rgt_x))

# VS: The fresnel convolution filter has a long length, so the input optical field (image) will be need to be padded to avoid artficats from circular convolutions.
# VS: We calculate padding required to input image in the case where the chirp convolution filter is sampled in the fourier domain.
def padding_sizes_fourier(N_y,N_x,pix_width,prop_dist,energy,mult_FWHM=1):
    return calc_padwid(N_y,N_x,pix_width,prop_dist,energy, mult_FWHM)

# VS: The fresnel convolution filter has a long length, so the input optical field (image) will be need to be padded to avoid artficats from circular convolutions.
# VS: We calculate padding required to input image in the case where the chirp convolution filter is sampled in the spatial domain.
def padding_sizes_spatial(N_y, N_x):
    # Filter length is same as source-plane (CT system) field-of-view
    fwidth=(N_y, N_x) #units - pixels
    # Dimensions after padding
    Npad_y = next_fft_len(N_y+fwidth[1])
    Npad_x = next_fft_len(N_x+fwidth[0])
    pad_y,pad_x = (Npad_y-N_y),(Npad_x-N_x)
    # LHS, RHS padding
    rgt_y,rgt_x = pad_y//2,pad_x//2
    lft_y,lft_x = pad_y-rgt_y,pad_x-rgt_x
    return ((lft_y,rgt_y),(lft_x,rgt_x))


def arr_pad(arr,pad_y,pad_x,mode,const=0):
    pad_width = [(0,0) for _ in range(arr.ndim-2)]
    pad_width.append((pad_y[0],pad_y[1]))
    pad_width.append((pad_x[0],pad_x[1]))
    if mode == 'constant':
        return np.pad(arr,pad_width=pad_width,mode=mode,constant_values=const)
    else:
        return np.pad(arr,pad_width=pad_width,mode=mode)

def arr_unpad(arr,pad_y,pad_x,ret_mask=False):
    if ret_mask is False:
        if arr.ndim==2:
            return arr[pad_y[0]:-pad_y[1],pad_x[0]:-pad_x[1]]
        elif arr.ndim==3:
            return arr[:,pad_y[0]:-pad_y[1],pad_x[0]:-pad_x[1]]
        elif arr.ndim==4:
            return arr[:,:,pad_y[0]:-pad_y[1],pad_x[0]:-pad_x[1]]
        else:
            print('Array shape is {}'.format(arr.shape))
            raise ValueError('Unsupported number of array dimensions')
    else:
        if arr.ndim==2:
            return np.s_[pad_y[0]:-pad_y[1],pad_x[0]:-pad_x[1]]
        elif arr.ndim==3:
            return np.s_[:,pad_y[0]:-pad_y[1],pad_x[0]:-pad_x[1]]
        elif arr.ndim==4:
            return np.s_[:,:,pad_y[0]:-pad_y[1],pad_x[0]:-pad_x[1]]
        else:
            raise ValueError('Unsupported number of array dimensions')
