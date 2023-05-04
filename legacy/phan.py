import numpy as np

def spheres_phantom(N_z,N_y,N_x,pix_width,cen_z=[0.0],cen_y=[0.0],cen_x=[0.0],radii=[None],delta=None,beta=None):
    assert len(cen_z)==len(cen_y) and len(cen_z)==len(cen_x) and len(cen_z)==len(radii),'Length of lists radii, cen_z, cen_y, and cen_x must be same'

    #Create a sphere phantom
    x = np.arange(0,N_x,1)*pix_width #x-axis coordinates
    y = np.arange(0,N_y,1)*pix_width #y-axis coordinates (x-y plane is parallel to rows of detector array)
    z = np.arange(0,N_z,1)*pix_width #z-axis coordinates (parallel to columns of detector array)
    z,y,x = np.meshgrid(z,y,x,indexing='ij') #Create a 3D dimensional grid
    delta_sphere = np.zeros((N_z,N_y,N_x),dtype=np.float32,order='C')
    beta_sphere = np.zeros((N_z,N_y,N_x),dtype=np.float32,order='C')

    if beta is None:
        beta = [1.0 for _ in range(len(radii))]
    if delta is None:
        delta = [1.0 for _ in range(len(radii))]
    for cz,cy,cx,rad,d,b in zip(cen_z,cen_y,cen_x,radii,delta,beta):
        rad = min(N_z,N_y,N_x)*pix_width/4 if rad is None else rad
        delta_sphere[(x-cx)**2+(y-cy)**2+(z-cz)**2<rad**2] = d
        beta_sphere[(x-cx)**2+(y-cy)**2+(z-cz)**2<rad**2] = b
 
    return delta_sphere,beta_sphere

def cylinder_phantom(N_z,N_y,N_x,pix_width,beta,delta,cen_y=[0.0],cen_x=[0.0],radii=[None]):
    assert len(cen_y)==len(cen_x) and len(cen_x)==len(radii),'Length of lists radii, cen_y, and cen_x must be same'

    #Create a sphere phantom
    x = np.arange(0,N_x,1)*pix_width #x-axis coordinates
    y = np.arange(0,N_y,1)*pix_width #y-axis coordinates (x-y plane is parallel to rows of detector array)
    z = np.arange(0,N_z,1)*pix_width #z-axis coordinates (parallel to columns of detector array)
    z,y,x = np.meshgrid(z,y,x,indexing='ij') #Create a 3D dimensional grid
    cylr = np.zeros((N_z,N_y,N_x),dtype=np.float32,order='C')

    for cy,cx,rad in zip(cen_y,cen_x,radii):
        rad = min(N_y,N_x)*pix_width/4 if rad is None else rad
        cylr[(x-cx)**2+(y-cy)**2<rad**2] = 1
 
    beta_vol = beta*cylr #Absorption index volume
    delta_vol = delta*cylr #Refractive index decrement volume
    return delta_vol,beta_vol 
