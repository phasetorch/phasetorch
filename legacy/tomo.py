import numpy as np
from skimage.transform import radon, iradon
from phasetorch.utils import get_wavelength


def sk_fbp(pix_width, num_views, ang_range, center, proj):
    angles = np.arange(0, num_views)*ang_range/num_views
    center = int(center)

    proj = np.swapaxes(proj, 0, 1)
    psh = proj.shape
    lsz, rsz = center, psh[-1]-center

    recon = np.zeros((psh[0], psh[-1], psh[-1]), dtype=np.float32, order='C')
    if lsz <= rsz:
        padw = rsz - lsz
        proj = np.pad(proj, ((0, 0), (0, 0), (padw, 0)), mode='edge')
        for i in range(psh[0]):
            rec = iradon(proj[i].T, theta=angles,
                         output_size=proj.shape[-1], circle=True)/pix_width
            recon[i] = rec[padw:, padw:]
    else:
        padw = lsz - rsz
        proj = np.pad(proj, ((0, 0), (0, 0), (0, padw)), mode='edge')
        for i in range(psh[0]):
            rec = iradon(proj[i].T, theta=angles,
                         output_size=proj.shape[-1], circle=True)/pix_width
            recon[i] = rec[:-padw, :-padw]

    return recon


def fbp_tomorec(pix_width, energy, num_views, ang_range, center, phase_proj=None, absorp_proj=None, backend='ltt', backend_options={'ltt_path': '/Users/mohan3/Desktop/Softw/LTT_v1.6.8/python'}):
    inv_wnum = get_wavelength(energy)/(2*np.pi)
    beta_vol, delta_vol = None, None
    if absorp_proj is not None:
        if backend == 'ltt':
            beta_vol = ltt_fbp(pix_width, num_views, ang_range, center,
                               absorp_proj, ltt_path=backend_options['ltt_path'])
        elif backend == 'scikit-image':
            beta_vol = sk_fbp(pix_width, num_views,
                              ang_range, center, absorp_proj)
        else:
            raise ValueError('backend must be either ltt or scikit-image')
        beta_vol = inv_wnum*beta_vol
    if phase_proj is not None:
        if backend == 'ltt':
            delta_vol = ltt_fbp(pix_width, num_views, ang_range, center,
                                phase_proj, ltt_path=backend_options['ltt_path'])
        elif backend == 'scikit-image':
            delta_vol = sk_fbp(pix_width, num_views,
                               ang_range, center, phase_proj)
        else:
            raise ValueError('backend must be either ltt or scikit-image')
        delta_vol = inv_wnum*delta_vol
    return delta_vol, beta_vol


def get_projs(pix_width, num_views, ang_range, center=None, beta_vol=None, delta_vol=None, backend='ltt', backend_options={'ltt_path': '/Users/mohan3/Desktop/Softw/LTT_v1.6.8/python'}):
    if center is None:
        center = delta_vol.shape[-1]/2

    if beta_vol is not None and delta_vol is not None:
        if beta_vol.shape != delta_vol.shape:
            raise ValueError("beta_vol and delta_vol must be of same shape")

    if delta_vol is not None:
        dsh = delta_vol.shape
        if dsh[1] != dsh[2]:
            raise ValueError("Last two dimensions of delta_vol must be same")
        if backend == 'ltt':
            delta_proj = ltt_fp(delta_vol, pix_width, num_views,
                                ang_range, center, ltt_path=backend_options['ltt_path'])
        elif backend == 'scikit-image':
            if center is not None:
                raise ValueError(
                    "scikit-image's radon() does not support arbitrary rotation centers")
            angles = np.arange(0, num_views, 1)*ang_range/num_views
            delta_proj = np.zeros(
                (dsh[0], len(angles), dsh[-1]), dtype=np.float32, order='C')
            for i in range(dsh[0]):
                delta_proj[i] = radon(delta_vol[i], angles, circle=True).T
            delta_proj = np.swapaxes(delta_proj, 0, 1)*pix_width
        else:
            raise ValueError('backend must be either ltt or scikit-image')

    if beta_vol is not None:
        bsh = beta_vol.shape
        if bsh[1] != bsh[2]:
            raise ValueError("Last two dimensions of beta_vol must be same")
        if backend == 'ltt':
            beta_proj = ltt_fp(beta_vol, pix_width, num_views, ang_range,
                               center, ltt_path=backend_options['ltt_path'])
        elif backend == 'scikit-image':
            if center is not None:
                raise ValueError(
                    "scikit-image's radon() does not support arbitrary rotation centers")
            angles = np.arange(0, num_views, 1)*ang_range/num_views
            beta_proj = np.zeros(
                (bsh[0], len(angles), bsh[-1]), dtype=np.float32, order='C')
            for i in range(bsh[0]):
                beta_proj[i] = radon(beta_vol[i], angles, circle=True).T
            beta_proj = np.swapaxes(beta_proj, 0, 1)*pix_width
        else:
            raise ValueError('backend must be either ltt or scikit-image')

    return beta_proj, delta_proj


def ltt_fp(vol, pix_width, num_views, ang_range, center, ltt_path):
    import sys
    sys.path.append(ltt_path)
    from LTTserver import LTTserver

    N_z = vol.shape[0]
    N_y = vol.shape[1]
    N_x = vol.shape[2]

    LTT = LTTserver()
    LTT.cmd("clearall")
    LTT.cmd("geometry = parallel")
    LTT.cmd("angularRange = {}".format(ang_range))
    LTT.cmd("diskIO = off")
    LTT.cmd("archdir = pwd")
    LTT.cmd("dataType = atten")

    LTT.cmd("nCols = {}".format(N_x))
    LTT.cmd("nRows = {}".format(N_z))
    LTT.cmd("nAngles = {}".format(num_views))

    LTT.cmd("pixelWidth = {}".format(pix_width))
    LTT.cmd("pixelHeight = pixelWidth")
    LTT.cmd("centerCol = {}".format(center))

    LTT.cmd("rzelements = " + str(N_z))
    LTT.cmd("ryelements = " + str(N_y))
    LTT.cmd("rxelements = " + str(N_x))
    LTT.cmd("softDefaultVolume")
    #LTT.cmd("windowFOV = False")

    LTT.setAllReconSlicesZ(vol.astype(np.float32))
    LTT.cmd("project")
    return LTT.getAllProjections().astype(np.float32)


def ltt_fbp(pix_width, num_views, ang_range, center, projs, ltt_path):
    import sys
    sys.path.append(ltt_path)
    from LTTserver import LTTserver

    N_z = projs.shape[1]
    N_y = projs.shape[2]
    N_x = N_y
    num_views = projs.shape[0]

    LTT = LTTserver()
    LTT.cmd("clearall")
    LTT.cmd("geometry = parallel")
    LTT.cmd("angularRange = {}".format(ang_range))
    LTT.cmd("diskIO = off")
    LTT.cmd("archdir = pwd")
    LTT.cmd("dataType = atten")

    LTT.cmd("nCols = {}".format(N_x))
    LTT.cmd("nRows = {}".format(N_z))
    LTT.cmd("nAngles = {}".format(num_views))

    LTT.cmd("pixelWidth = {}".format(pix_width))
    LTT.cmd("pixelHeight = pixelWidth")
    LTT.cmd("centerCol = {}".format(center))

    LTT.cmd("rzelements = " + str(N_z))
    LTT.cmd("ryelements = " + str(N_y))
    LTT.cmd("rxelements = " + str(N_x))
    LTT.cmd("softDefaultVolume")
    #LTT.cmd("windowFOV = False")

    LTT.setAllProjections(projs.astype(np.float32))
    LTT.cmd("FBP")
    return LTT.getAllReconSlicesZ().astype(np.float32)
