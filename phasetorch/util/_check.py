from phasetorch.util._pad import calc_padwid
import numpy as np
#from numpy.distutils.misc_util import is_sequence
import collections

def get_Ndists(dists):
#    if is_sequence(dists):
#        dists = np.array(dists, dtype=np.double, order='C')

    # https://stackoverflow.com/questions/2937114/python-check-if-an-object-is-a-sequence
    # https://stackoverflow.com/questions/16807011/python-how-to-identify-if-a-variable-is-an-array-or-a-scalar
    if isinstance(dists, str):
        raise ValueError("dists cannot be a string")    
    elif isinstance(dists, collections.abc.Sequence):
        dists = np.array(dists, dtype=np.double, order='C')
    elif not isinstance(dists, np.ndarray):
        dists = np.array([dists], dtype=np.double, order='C')
   
    print('Dists is {}'.format(dists)) 
    return len(dists), dists 

def assert_Ndists(arr, N_dists, dim=None):
    if N_dists == 1 and not isinstance(arr,np.ndarray):
        assert not hasattr(type(arr), '__iter__'),'Array must not be a sequence type'
    else:
        assert arr.shape[dim]==N_dists,'Dimension {} of array must be of size {}'.format(dim,N_dists)

def assert_dims(arr, dmin, dmax, dlast=None):
    if dlast is not None:
        assert arr.shape[-1]==dlast,'Last dimension of array must be of size {}'.format(dlast)
    assert arr.ndim>=dmin,'data must have a minimum of {} dimensions'.format(dmin)
    assert arr.ndim<=dmax,'data must have a maximum of {} dimensions'.format(dmax)
    
def expand_dims(arr, dmax, dapp):
    if arr.ndim <= dmax:
        arr = np.expand_dims(arr, axis=dapp)
    return arr

def multi_chkpad(rads, dists, pix_wid, energy, mult_FWHM, weights=None, add_dim=1):
    if weights is not None:
        assert weights.shape==rads.shape       

    assert add_dim==1 or add_dim==0
    if len(rads.shape)==3:
        rads = rads[:,np.newaxis] if add_dim==1 else rads[np.newaxis]
        if weights is not None:
            weights = weights[:,np.newaxis] if add_dim==1 else weights[np.newaxis] 
    elif len(rads.shape)!=4:
        raise ValueError('ERROR: rads must have either 3 or 4 dimensions')
    
    if not isinstance(dists, list):
        dists = float(dists)
        dists = [dists]
 
    dists = np.array(dists) 
    N_y,N_x = rads.shape[-2],rads.shape[-1] 
    pad_y,pad_x = calc_padwid(N_y,N_x,pix_wid,max(dists),energy,mult_FWHM)
    
    assert rads.shape[1]==len(dists),'Second dimension of rads must equal the length of the list dists'
    rads = np.pad(rads, pad_width=((0,0),(0,0),(pad_y[0],pad_y[1]),(pad_x[0],pad_x[1])), mode='edge')
    print('Padded size is {}x{}'.format(rads.shape[-2],rads.shape[-1]))

    if weights is not None:
        weights = np.pad(weights, pad_width=((0,0),(0,0),(pad_y[0],pad_y[1]),(pad_x[0],pad_x[1])), mode='constant', constant_values=0.0)
        return rads,weights,dists,pad_y,pad_x
    else:
        return rads,dists,pad_y,pad_x
    

def single_chkpad(rads, dist, pix_wid, energy, mult_FWHM):
    if len(rads.shape)==2:
        rads = rads[np.newaxis]
    elif len(rads.shape)!=3:
        raise ValueError('ERROR: rads must have either 2 or 3 dimensions')
    
    N_y,N_x = rads.shape[-2],rads.shape[-1] 
    pad_y,pad_x = calc_padwid(N_y,N_x,pix_wid,dist,energy,mult_FWHM)
    
    rads = np.pad(rads, pad_width=((0,0),(pad_y[0],pad_y[1]),(pad_x[0],pad_x[1])), mode='edge')
    print('Padded size is {}x{}'.format(rads.shape[-2],rads.shape[-1]))
    return rads,dist,pad_y,pad_x

 
