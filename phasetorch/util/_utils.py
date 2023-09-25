import numpy as np
import datetime
#import phasect._clib.lib as cl

# Constants
# Js or m^2kg/s. Obtained from google, please verify from scientific literature.
h = 6.62607004*(10**-34)
# m/s. Obtained from google, please verify from scientific literature.
c = 299792458
# Obtained from google, please verify from scientific literature.
JperkeV = 1.60218*(10**-16)

"""
some more physical constants
added by SS @02/16/2022
"""
# some physical constants
cAvogadro = 6.02214076E23  # Avogadro's constant Na
cBoltzmann = 1.380649E-23  # Boltzmann's constant, K
cCharge = 1.602176634E-19  # charge of electron
cJ2eV = 1.602176565E-19  # joule to ev, JperkeV*1e-3
cLight = 299792458.0  # speed of light, same as c above but name is more descriptive
cMoment = 9.2740100707E-24  # magnetic moment of electron
cPermea = 1.2566370616E-6  # permeability of free space
cPermit = 8.8541878163E-12  # permittivity of free space
cPlanck = 6.62607015E-34  # same as h above but name is more descriptive
cRestmass = 9.1093837090E-31  # rest mass of electron
cClassicalelectronRad = 2.8179403e-6  # classical electron radius in nm


def get_wavelength(energy):
    """
    Compute the wavelength of X-rays from its energy.

    Parameters
    ----------
    energy : float
        Energy of X-rays (units of keV).

    Returns
    -------
    wavelength : float
        Wavelength of X-rays in millimeters (mm).
    """

    # return cl.get_wavelength(energy)
    return (cPlanck*cLight/(energy*cJ2eV))  # in mm - by SS
    #return (h*c/(energy*JperkeV))*(10**3) #in mm - by AM/JB

"""
simple function to convert from beam energy in kev to nm
more useful for refractive index computation
added by SS @02/16/2022
"""


def keV2nm(x):
    return get_wavelength(x) * 1E6


def cphase_cproj(arr, energy):
    """
    Parameters
    ----------
    arr : numpy.ndarray
        Complex or real valued phase/absorption images.
    energy : float
        X-ray energy in units of keV.
    
    Returns
    -------
    proj : numpy.ndarray
        Complex or real projection of delta/beta.
    """

    wlength = get_wavelength(energy)
    return arr*wlength/(2*np.pi)


def cproj_cphase(arr, energy):
    """
    Parameters
    ----------
    arr : numpy.ndarray
        Complex or real valued delta/beta images.
    energy : float
        X-ray energy in units of keV.

    Returns
    -------
    cphase : numpy.ndarray
        Complex or real valued phase/absorption images.
    """

    wlength = get_wavelength(energy)
    return arr*(2*np.pi)/wlength


def get_fft(image, pix_width):
    return np.fft.fft2(image, norm='ortho')


def get_freqs(image_shape, pix_width):
    fx = np.fft.fftfreq(image_shape[1], d=pix_width)
    fy = np.fft.fftfreq(image_shape[0], d=pix_width)
    fx_mesh, fy_mesh = np.meshgrid(fx, fy)
    return fx_mesh, fy_mesh


def get_inv_fft(image):
    return np.fft.ifft2(image, norm='ortho')


def get_window(window, num_rows, num_cols):
    if window == 'rectangular':
        win_y = np.ones(num_rows)
        win_x = np.ones(num_cols)
    elif window == 'hanning':
        win_y = np.hanning(
            num_rows+1)[:-1] if num_rows % 2 == 0 else np.hanning(num_rows)
        win_x = np.hanning(
            num_cols+1)[:-1] if num_cols % 2 == 0 else np.hanning(num_cols)
    elif window == 'hamming':
        win_y = np.hamming(
            num_rows+1)[:-1] if num_rows % 2 == 0 else np.hamming(num_rows)
        win_x = np.hamming(
            num_cols+1)[:-1] if num_cols % 2 == 0 else np.hamming(num_cols)
    elif window == 'bartlett':
        win_y = np.bartlett(
            num_rows+1)[:-1] if num_rows % 2 == 0 else np.bartlett(num_rows)
        win_x = np.bartlett(
            num_cols+1)[:-1] if num_cols % 2 == 0 else np.bartlett(num_cols)
    elif window == 'blackman':
        win_y = np.blackman(
            num_rows+1)[:-1] if num_rows % 2 == 0 else np.blackman(num_rows)
        win_x = np.blackman(
            num_cols+1)[:-1] if num_cols % 2 == 0 else np.blackman(num_cols)
    else:
        raise ValueError(
            'Only rectangular, hanning, hamming, bartlett, and backman windows are supported')
    win_np = np.outer(win_y, win_x)
    win_np = np.fft.ifftshift(win_np).astype(np.double, order='C')
    assert win_np[0, 0] == np.max(
        win_np), 'Maximum value of window function must be at (0,0)'
    assert win_np[0, 0] == 1
    return win_np


def tsor2img(tsor, fpath):
    from PIL import Image
    tsor = tsor.detach().cpu().numpy()
    assert tsor.ndim == 2
    img = Image.fromarray(tsor)
    img.save(fpath)

def time_stamp():
    now = datetime.datetime.now()
    return now.strftime('%y%m%d_%H%M%S')

