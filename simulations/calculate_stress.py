import sys
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt


for i in range(len(sys.argv)):
    if (sys.argv[i].find("-n")==0):
        slip_file  = sys.argv[i+1]
        mud        = float(sys.argv[i+2])
        S3	   = float(sys.argv[i+3])
        S          = float(sys.argv[i+4])
        dc         = float(sys.argv[i+5])
        mu         = float(sys.argv[i+6])


def ll2xy(Lon, Lat, Lon_ref, Lat_ref):
    Earth_radius = 6371.0
    Lon_scale = np.cos(Lat_ref/180.0*3.1415926)*3.1415926*Earth_radius/180.0
    Lat_scale = 3.1415926*Earth_radius/180.0
    X         = (Lon-(Lon_ref))*Lon_scale
    Y         = (Lat-(Lat_ref))*Lat_scale
    return X,Y


def rotate_2d_about_point(uxp, uyp, uzp, pivot_xy, angle_deg):
    """
    Input:
        uxp, uyp, uzp, uv_ext - Original data (same dimensions)
        pivot_xy              - XY coordinates of the rotation pivot point [x0, y0]
        angle_deg             - Rotation angle (in degrees)
    Output:
        uxp_rot, uyp_rot      - Rotated XY coordinates (Z and UV remain unchanged)
    """
    # Convert to radians
    theta = np.radians(angle_deg)
    # Extract the rotation center
    x0 = pivot_xy[0]
    y0 = pivot_xy[1]
    # Translate to have the rotation center as the origin
    uxp_centered = uxp - x0
    uyp_centered = uyp - y0
    # Apply the 2D rotation matrix
    uxp_rot = uxp_centered * np.cos(theta) - uyp_centered * np.sin(theta) + x0
    uyp_rot = uxp_centered * np.sin(theta) + uyp_centered * np.cos(theta) + y0
    # Z and color values remain unchanged
    uzp_rot = uzp
    return uxp_rot, uyp_rot, uzp_rot

def find_nearest_point(uxp, uyp, uzp, uv_ext, x0, y0):
    """
    Find the nearest point in the grid to a given target point (x0, y0).
    Parameters:
        uxp: Array of x-coordinates
        uyp: Array of y-coordinates
        uzp: Array of z-coordinates (not used in the calculation here)
        uv_ext: Additional data (not used in the calculation here)
        x0: Target x-coordinate
        y0: Target y-coordinate

    Returns:
        linear_idx: Linear index of the nearest point
        min_uy: y-coordinate of the nearest point
        min_ux: x-coordinate of the nearest point
    """
    # Input parameter checks
    if uxp.size == 0 or uyp.size == 0 or uzp.size == 0 or uv_ext.size == 0:
        raise ValueError('Input arrays cannot be empty!')
    if not (uxp.shape == uyp.shape == uzp.shape == uv_ext.shape):
        raise ValueError('Input arrays must have the same dimensions!')
    # Calculate squared distances from the target point
    distances_sq = (uxp - x0) ** 2 + (uyp - y0) ** 2
    # Find the minimum distance's linear index
    min_dist = np.min(distances_sq)
    linear_idx = np.argmin(distances_sq)
    if linear_idx is None or np.isnan(min_dist):
        raise ValueError('Unable to find a valid nearest point, check input data!')
    # Check if linear index is out of bounds (not strictly necessary for numpy)
    max_index = uxp.size
#    if linear_idx < 0 or linear_idx >= max_index:
#        raise IndexError(f'Linear index out of bounds: index={linear_idx}, max allowed={max_index}')
    # Extract values of the nearest point
    min_uy = uyp.flatten()[linear_idx]  # Using flatten to handle indexing
    min_ux = uxp.flatten()[linear_idx]
    return linear_idx, min_uy, min_ux


# Assuming strike is along the X direction
def cal_tau(X,Y,dep,slip,rake,dip,mu,dx,dz):
    from okada_wrapper import dc3d0wrapper, dc3dwrapper
    num_p    =  slip.shape[0]
    slip_str =  slip[:]*np.cos(rake/180.0*np.pi)
    slip_dip =  slip[:]*np.sin(rake/180.0*np.pi)

    tau_str = np.zeros((num_p))
    tau_dip = np.zeros((num_p))
    tau_nor = np.zeros((num_p))

    lamb   = mu
    alpha  = (lamb + mu) / (lamb + 2 * mu)
    n1     = [0.0,-1.0,0.0]
    s1     = [1.0,0.0,0.0]
    d1     = [0.0,0.0,1.0]
    uij    = np.zeros((num_p,3,3))

    # The observation point
    for i in range(num_p): 
        # The slip patch
        for j in range(num_p):
            success,u,grad_u = dc3dwrapper(alpha, [(X[i]-X[j])*1e3,(Y[i]-Y[j])*1e3,dep[i]*1e3],
                                           -dep[j]*1e3, dip,
                                       [-dx*1e3/2.0,dx*1e3/2.0],[-dz*1e3/2.0,dz*1e3/2.0],[slip_str[j],slip_dip[j],0.0])
            assert(success == 0)
            uij[i,:,:] = uij[i,:,:] + grad_u[:,:]

        S11 = lamb*(uij[i,0,0]+uij[i,1,1]+uij[i,2,2]) + 2*mu*uij[i,0,0]
        S22 = lamb*(uij[i,0,0]+uij[i,1,1]+uij[i,2,2]) + 2*mu*uij[i,1,1]
        S33 = lamb*(uij[i,0,0]+uij[i,1,1]+uij[i,2,2]) + 2*mu*uij[i,2,2]
        S12 = mu  *(uij[i,0,1]+uij[i,1,0])
        S23 = mu  *(uij[i,1,2]+uij[i,2,1])
        S13 = mu  *(uij[i,0,2]+uij[i,2,0])

        T1  = S11*n1[0] + S12*n1[1] + S13*n1[2]
        T2  = S12*n1[0] + S22*n1[1] + S23*n1[2]
        T3  = S13*n1[0] + S23*n1[1] + S33*n1[2]

        tau_str[i]  =  -(T1*s1[0] + T2*s1[1] + T3*s1[2])
        tau_dip[i]  =  -(T1*d1[0] + T2*d1[1] + T3*d1[2])
        tau_nor[i]  =   (T1*n1[0] + T2*n1[1] + T3*n1[2])
    return tau_str,tau_dip,tau_nor


#####################################################
#####################################################


km2m   = 1e3
GPa2Pa = 1e9  # GPa to Pa
mu     = mu * GPa2Pa  # shear modulus

Lon_ref=96.0442
Lat_ref=21.9924

#Read the kinematic slip model
#x y depth yinc xinc strike dip rake slip Sample
dis= pd.read_csv(slip_file, sep=r'\s+')
[X,Y]  = ll2xy(dis['x'], dis['y'], Lon_ref, Lat_ref)
Dep    = - dis['depth'] 
slip   = dis['slip']
dip    = np.mean(dis['dip'])    # Use the average number
rake   = np.abs(dis['rake'])
xinc   = np.mean(dis['xinc'])
yinc   = np.mean(dis['yinc'])

# Truncate the slip for those smaller than a value
truncate_slip = 0.0
#slip = np.where(slip < truncate_slip, 0, slip)

# The rotation angle is 270-strike
#[X_rot,Y_rot,Dep] = rotate_2d_about_point(X, Y, Dep, [0,0], strike-270)
dx_str   = ( xinc**2+yinc**2)**0.5
dz_dip   =  (np.unique(Dep)[1] - np.unique(Dep)[0]) / np.sin(np.radians(dip))

[tau_str,tau_dip,tau_nor] = cal_tau(X,Y,Dep,slip,rake,dip,mu,dx_str,dz_dip)

tau_str = tau_str + np.cos(np.deg2rad(rake))*mud*S3
tau_dip = tau_dip + np.sin(np.deg2rad(rake))*mud*S3
tau     = (tau_str**2+tau_dip**2)**0.5

# Use griddata to interpolate the data
grid_y, grid_z = np.mgrid[np.min(Y):np.max(Y):100j, np.min(Dep):np.max(Dep):100j]
grid_slip      = griddata(np.column_stack((Y, Dep)), slip,    (grid_y, grid_z), method='linear',fill_value=0)
grid_tau_str   = griddata(np.column_stack((Y, Dep)), tau_str, (grid_y, grid_z), method='linear',fill_value=0)
grid_tau_dip   = griddata(np.column_stack((Y, Dep)), tau_dip, (grid_y, grid_z), method='linear',fill_value=0)
grid_tau       = griddata(np.column_stack((Y, Dep)), tau, (grid_y, grid_z), method='linear',fill_value=99999e6)
grid_x         = griddata(np.column_stack((Y, Dep)),       X, (grid_y, grid_z), method='nearest')

grid_mus  = ((1+S)*(grid_tau-mud*S3)/(S3) + mud)
upbound   = np.where(grid_y <= np.min(Y))
lowbound  = np.where(grid_y >= np.max(Y))
mask = (grid_y >= 80) | (grid_y <= -410)
grid_tau_str[mask] = 0
grid_tau_dip[mask] = 0
grid_tau[mask] = 99999e6


### Plot
#plt.figure(figsize=(10, 8))
#
#plt.subplot(3,2,1)
#plt.imshow(grid_slip.T, extent=(np.min(grid_x), np.max(grid_x), np.min(grid_z), np.max(grid_z)), origin='lower', cmap='hot_r')
#plt.title('Slip')
#plt.xlabel('X Coordinate')
#plt.ylabel('Y Coordinate')
#cbar = plt.colorbar(shrink=0.2)
#cbar.set_label('Slip (m)')
#
#plt.subplot(3,2,2)
#plt.imshow(grid_tau_str.T / 1e6, extent=(np.min(grid_x), np.max(grid_x), np.min(grid_z), np.max(grid_z)), origin='lower', cmap='hot_r')
#plt.title('Along-strike stress drop')
#plt.xlabel('X Coordinate')
#plt.ylabel('Y Coordinate')
#cbar = plt.colorbar(shrink=0.2)
#cbar.set_label('Stress drop (MPa)')
#
#plt.subplot(3,2,3)
#plt.imshow(grid_tau_dip.T / 1e6, extent=(np.min(grid_x), np.max(grid_x), np.min(grid_z), np.max(grid_z)), origin='lower', cmap='hot_r')
#plt.title('Along-dip stress drop')
#plt.xlabel('X Coordinate')
#plt.ylabel('Y Coordinate')
#cbar = plt.colorbar(shrink=0.2)
#cbar.set_label('Stress drop (MPa)')
#
#plt.subplot(3,2,4)
#plt.imshow(grid_tau.T / 1e6, extent=(np.min(grid_x), np.max(grid_x), np.min(grid_z), np.max(grid_z)), origin='lower', cmap='hot_r')
#plt.title('Total stress drop')
#plt.xlabel('X Coordinate')
#plt.ylabel('Y Coordinate')
#cbar = plt.colorbar(shrink=0.2)
#cbar.set_label('Stress drop (MPa)')
#
#plt.subplot(3,2,5)
#plt.scatter(X,Y)
#
#plt.tight_layout()
#plt.savefig('stress_and_slip.png')
#
############


output= open("./DATA/initial_tau_str.dat","w")
output.writelines(str(grid_x.shape[0]*grid_x.shape[1]))
output.writelines("\n")
for i in range(grid_x.shape[0]):
    for j in range(grid_x.shape[1]):
        output.writelines(str(grid_x[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(grid_y[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(grid_z[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(grid_tau_str[i,j]))    
        output.writelines("\n")
output.close()

output= open("./DATA/initial_tau_dip.dat","w")
output.writelines(str(grid_x.shape[0]*grid_x.shape[1]))
output.writelines("\n")
for i in range(grid_x.shape[0]):
    for j in range(grid_x.shape[1]):
        output.writelines(str(grid_x[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(grid_y[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(grid_z[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(grid_tau_dip[i,j]))
        output.writelines("\n")
output.close()

output= open("./DATA/initial_mus.dat","w")
output.writelines(str(grid_x.shape[0]*grid_x.shape[1]))
output.writelines("\n")
for i in range(grid_x.shape[0]):
    for j in range(grid_x.shape[1]):
        output.writelines(str(grid_x[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(grid_y[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(grid_z[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(grid_mus[i,j]))
        output.writelines("\n")
output.close()


output= open("./DATA/initial_dc.dat","w")
output.writelines(str(grid_x.shape[0]*grid_x.shape[1]))
output.writelines("\n")
for i in range(grid_x.shape[0]):
    for j in range(grid_x.shape[1]):
        output.writelines(str(grid_x[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(grid_y[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(grid_z[i,j]*1e3))
        output.writelines("  ")
        output.writelines(str(dc)) 
        output.writelines("\n")
output.close()
