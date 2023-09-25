import numpy as np

def path_lengths_cylinder(N_y, N_x, pix_width, center_x, radius):
    """
    Return the path length through a cylinder.

    Parameters
    ----------
    N_y : int
        Number of pixels along the vertical direction of a radiograph (y-axis).
    N_x : int
        Number of pixels along the horizontal direction of a radiograph (x-axis).
    pix_width : float
        Pixel width in units of mm.
    center_x : float
        Center of cylinder along the horizontal direction in units of mm.
    radius : float
        Radius of cylinder in units of mm.

    Returns
    -------
    projs : numpy.ndarray
        Array of X-ray path lengths in units of mm.
    mask : numpy.ndarray
        Mask to indicate the location of non-zero projections.
    """

    x = np.arange(0, N_x, 1)*pix_width
    y = np.arange(0, N_y, 1)*pix_width
    y, x = np.meshgrid(y, x, indexing="ij")

    # Ground-truth projections. Unitary dimension indicates one view.
    projs = np.zeros_like(x)
    # Mask for non-zero projections of the simulated cylinder.
    mask = (radius**2 - (x-center_x)**2) > 0
    # Generate path lengths along the X-ray propagation direction.
    projs[mask] = 2*np.sqrt((radius**2 - (x-center_x)**2)[mask])
    
    return projs, mask

def path_lengths_sphere(N_y, N_x, pix_width, center_y, center_x, radius):
    """
    Return the path length through a sphere.

    Parameters
    ----------
    N_y : int
        Number of pixels along the vertical direction of a radiograph (y-axis).
    N_x : int
        Number of pixels along the horizontal direction of a radiograph (x-axis).
    pix_width : float
        Pixel width in units of mm.
    center_y : float
        Center of sphere along the vertical direction in units of mm.
    center_x : float
        Center of sphere along the horizontal direction in units of mm.
    radius : float
        Radius of sphere in units of mm.

    Returns
    -------
    projs : numpy.ndarray
        Array of X-ray path lengths in units of mm.
    mask : numpy.ndarray
        Mask to indicate the location of non-zero projections.
    """

    x = np.arange(0, N_x, 1)*pix_width
    y = np.arange(0, N_y, 1)*pix_width
    y, x = np.meshgrid(y, x, indexing="ij")

    # Ground-truth projections. Unitary dimension indicates one view.
    projs = np.zeros_like(x)

    # Mask for non-zero projections of the simulated sphere.
    planar_dists2 = (x-center_x)**2 + (y-center_y)**2
    mask = (radius**2 - planar_dists2) > 0

    # Generate path lengths along the X-ray propagation direction.
    projs[mask] = 2*np.sqrt((radius**2 - planar_dists2)[mask])
    
    return projs, mask
