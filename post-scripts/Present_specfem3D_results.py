#from scipy.io import netcdf
import numpy as np
import sys
import os
import struct
import itertools
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d


Model_dir = None
Model_name = None
found_n = False

for i in range(1, len(sys.argv)):
    if sys.argv[i] == "-n":
        if i + 2 >= len(sys.argv):
            print("ERROR：'-n' need two parameters Model_dir and Model_name")
            sys.exit(1)
        Model_dir = sys.argv[i + 1]
        Model_name = sys.argv[i + 2]
        found_n = True
        break  

# Moment rate path
output_dir = "./ps"
# Work directory
# All the Specfem3D results are saved in the directory named as Model_name
# Grid size used to intepolate the fault nodes
grid_size            = 0.2  # in km
# display the fault horizontally or vertically
display_horizontally = False
#True
# if display_horizontally = False, then display vertically. Rotate=0 means projecting along X-Z plane, and Rotate=90 means projecting along Y-Z plane.
Rotate  =  90.0
# The S wave speed, used to normalized the rupture speed.
Vs = 3.33   # in km/s
# Number of countours of rupture front
ContourT = 5
# Number of countours of final slip
ContourS = 3
# Python version =2 or >3
Python_version = 3.8

#=====================================


## Files check
file_dir  = Model_dir + "/" +Model_name
if not os.path.isdir(file_dir):
    print("The directory that contains the data doesn't exist...")
    exit()
file_list = os.listdir(file_dir)
snap_list = []
time_list = []
for name in file_list:
    if(name.find('Snapshot')==0):
        snap_list.append(name)
        time_list.append(int(name.split('Snapshot')[1].split('_')[0]))
num_data   = len(time_list)
time_list  = np.asarray(sorted(time_list))
print("") 
print( "The model name is:", Model_name)
print( "The model has", num_data, "data files.")
print( "The time list is:", time_list)
print("")
if(display_horizontally):
    print("Project the fault horizontally.")
else:
    print("Project the fault onto a vertical plane (clockwisely rotated ", Rotate, " degree about X-Z plane).")
print("")

#==================
#   Functions   ===
#==================
def  FSEM3D_snapshot(filename):
    data = type("",(),{})()
    NDAT    = 14
    length  = 4
    binary_file = open(filename,"rb")
    BinRead = []
    for ii in range(NDAT):
        read_buf = binary_file.read(4)    # the number of bytes of int is 4
        number = struct.unpack('1i',read_buf)[0]
        if(Python_version<3):
           N = number/length
        else:
           N = number//length
        read_buf = binary_file.read(number)
        read_d   = struct.unpack(str(N)+'f',read_buf)
        read_d   = np.array(read_d)
        BinRead.append(read_d)
        read_buf = binary_file.read(4)    # the number of bytes of int is 4
        number = struct.unpack('1i',read_buf)[0]
    data.X  = BinRead[0]/1e3   # in km
    data.Y  = BinRead[1]/1e3  # in km
    data.Z  = BinRead[2]/1e3  # in km
    data.Dx = BinRead[3]
    data.Dz = BinRead[4]
    data.Vx = BinRead[5]
    data.Vz = BinRead[6]
    data.Tx = BinRead[7]      # in MPa
    data.Ty = BinRead[8]
    data.Tz = BinRead[9]     # in MPa
    data.S  = BinRead[10]
    data.Sg = BinRead[11]     # in MPa
    data.Trup = BinRead[12]
    data.Tpz = BinRead[13]
    return data

Data = []
for ii in range(num_data):
    name = file_dir+"/Snapshot"+str(time_list[ii])+"_F1.bin"
    Data.append(FSEM3D_snapshot(name))
X_Y_Z    = np.column_stack((Data[0].X,Data[0].Y,Data[0].Z))


# Create the grid arrays of nodes
X    = Data[0].X   
Y    = Data[0].Y
Z    = Data[0].Z   
# if it is a vertical fault (along X or Y axis)
if((np.min(X) == np.max(X) or np.min(Y)==np.max(Y)) and display_horizontally):
    print( "Warning: it is a vertical fault along X or Y axis, the fault will be displayed vertically!!!")
    print( "Please change display_horizontally to be False and setup the strike of fault to be 0 or 90.")
    exit()
if(not display_horizontally):
    X_rot = X * np.cos(Rotate/180.0*np.pi)   + Y * np.sin(Rotate/180.0*np.pi)
    Y_rot = X * - np.sin(Rotate/180.0*np.pi) + Y * np.cos(Rotate/180.0*np.pi)
    X     = X_rot
    Y     = Y_rot

[X_lower,X_upper,Y_lower,Y_upper,Z_lower,Z_upper] = [np.min(X),np.max(X),np.min(Y),np.max(Y),np.min(Z),np.max(Z)]
X_dim = int((X_upper-X_lower)/grid_size+1)
Y_dim = int((Y_upper-Y_lower)/grid_size+1)
Z_dim = int((Z_upper-Z_lower)/grid_size+1)

if(display_horizontally):
    X_Y    = np.column_stack((X, Y))
    Relief = Z
    grid_x, grid_y = np.mgrid[X_lower:X_upper:X_dim*1j, Y_lower:Y_upper:Y_dim*1j]
    fault_range =  [X_lower,X_upper,Y_lower,Y_upper]
    fault_dims  = [X_dim,Y_dim]
else:
    X_Y    = np.column_stack((X, Z))
    Relief = X
    grid_x, grid_y = np.mgrid[X_lower:X_upper:X_dim*1j, Z_lower:Z_upper:Z_dim*1j]
    fault_range =  [X_lower,X_upper,Z_lower,Z_upper]
    fault_dims  = [X_dim,Z_dim]



# The relief of the axis that is normal to the projected surface
Z_grid         = griddata(X_Y[:,0:2], Relief, (grid_x,grid_y), method='linear')

X_gradient,Y_gradient  =  np.gradient(Z_grid)
Nor_dir = np.zeros((X_gradient.shape[0],X_gradient.shape[1],3))
for i in range(fault_dims[0]):
    for j in range(fault_dims[1]):
         Nor_dir[i,j,:] = np.cross([1,0,X_gradient[i,j]], [0,1,Y_gradient[i,j]])

# Rupture time
Final_rup_time = Data[len(Data)-1].Trup
Init_t0 = griddata(X_Y[:,0:2], Final_rup_time, (grid_x,grid_y), method='linear')

# Rupture speed and direction
vr      = np.zeros((Init_t0.shape))
vr_dir  = np.zeros((Init_t0.shape))
for x in range(fault_dims[0]):
    for y in range(fault_dims[1]):
        # Skip the boundaries and unbroken nodes
        if(x==0 or x==fault_dims[0]-1 or y==0 or y==fault_dims[1]-1 or np.isnan(Init_t0[x,y])):
            vr[x,y] = np.nan
            continue
        # Calculate the gradient of rupture time
        delta_x = (Init_t0[x+1,y+1]-Init_t0[x-1,y+1]+Init_t0[x+1,y-1]-Init_t0[x-1,y-1]) / 4.0 / ((grid_size*X_gradient[x,y])**2 + grid_size**2)**0.5 
        delta_y = (Init_t0[x+1,y+1]-Init_t0[x+1,y-1]+Init_t0[x-1,y+1]-Init_t0[x-1,y-1]) / 4.0 / ((grid_size*Y_gradient[x,y])**2 + grid_size**2)**0.5 
        # Calculate rupture speed and direction
        if (np.abs(delta_x)<1e-6 and np.abs(delta_x)<1e-6):
            vr[x,y] = np.nan
            vr_dir[x,y]  = np.nan
        else:
            vr[x,y] = 1 / (delta_x**2 + delta_y**2)**0.5 / Vs
            vr_dir[x,y]  =  delta_x / (delta_x**2 + delta_y**2)**0.5


# Final slip
Final_slip_str = Data[len(Data)-1].Dx 
Final_slip_dip = Data[len(Data)-1].Dz
Final_slip     = (Final_slip_str**2 + Final_slip_dip**2)**0.5
Slip_grid      = griddata(X_Y[:,0:2], Final_slip, (grid_x,grid_y), method='linear')
# Stress drop
Stress_drop_str = Data[0].Tx/1e6 - Data[len(Data)-1].Tx/1e6 
Stress_drop_dip = Data[0].Ty/1e6 - Data[len(Data)-1].Ty/1e6 
Stress_drop_nor = Data[len(Data)-1].Tz/1e6
str_grid        = griddata(X_Y[:,0:2], Stress_drop_str, (grid_x,grid_y), method='linear')
dip_grid        = griddata(X_Y[:,0:2], Stress_drop_dip, (grid_x,grid_y), method='linear')
nor_grid        = griddata(X_Y[:,0:2], Stress_drop_nor, (grid_x,grid_y), method='linear')
# Initial stress
initial_stress_x = Data[0].Tx/1e6
initial_stress_y= Data[0].Ty/1e6 
stress_x_grid = griddata(X_Y[:,0:2], initial_stress_x, (grid_x, grid_y), method='linear')  
stress_y_grid = griddata(X_Y[:,0:2], initial_stress_y, (grid_x, grid_y), method='linear')
total_stress_grid = np.sqrt(stress_x_grid**2 +  stress_y_grid**2 )


###########  Output data
output= open("data/"+Model_name+"-results.dat","w")
for x in range(fault_dims[0]):
    for y in range(fault_dims[1]):
        output.writelines(str(grid_x[x,y]))
        output.writelines("  ")
        output.writelines(str(grid_y[x,y]))
        output.writelines("  ")
        output.writelines(str(Init_t0[x,y]))
        output.writelines("  ")
        output.writelines(str(vr[x,y]))
        output.writelines("  ")
        output.writelines(str(Slip_grid[x,y]))
        output.writelines("  ")
        output.writelines(str(str_grid[x,y]))
        output.writelines("  ")
        output.writelines(str(dip_grid[x,y]))
        output.writelines("  ")
        output.writelines(str(stress_x_grid[x,y]))
        output.writelines("\n")
output.close()

print("fault data is saved.")










##########################
### Plot figure    #######
##########################
fig = plt.figure(figsize=(10, 12))  # Figure size
plt.subplots_adjust(hspace=0.5)     # Space between subplots

# Rupture time
ax1 = plt.subplot2grid((6, 1), (0, 0), colspan=1, rowspan=1)
im1 = ax1.imshow(Init_t0.T, extent=fault_range, origin='lower', cmap='hot_r', aspect='auto')
cs1 = ax1.contour(grid_x, grid_y, Init_t0, ContourT, colors='k', linewidths=0.1)
cbar1 = plt.colorbar(im1, ax=ax1, extend='both', shrink=0.8, ticks=[0,10,20,30,40])
cbar1.set_label('Time (s)')
ax1.set_title('Rupture time')
ax1.set_xlabel('Depth (km)') 
ax1.set_ylabel('Along-strike distance (km)') 

# Initial stress
ax2 = plt.subplot2grid((6, 1), (1, 0), colspan=1, rowspan=1)
im2 = ax2.imshow(stress_x_grid.T, extent=fault_range, origin='lower', cmap='hot_r', aspect='auto', vmin=40, vmax=np.nanmax(total_stress_grid))
cbar2 = plt.colorbar(im2, ax=ax2, extend='both', shrink=0.8, ticks=[40,50,60])
cbar2.set_label('Stress (MPa)')
ax2.set_title('Initial Stress')
ax2.set_xlabel('Depth (km)') 
ax2.set_ylabel('Along-strike distance (km)') 

# Rupture speed
ax3 = plt.subplot2grid((6, 1), (2, 0), colspan=1, rowspan=1)
im3 = ax3.imshow(vr.T, extent=fault_range, origin='lower', cmap='hot_r', vmin=0, vmax=1.732, aspect='auto')
cs3 = ax3.contour(grid_x, grid_y, Init_t0, ContourT, colors='k', linewidths=0.1)
cbar3 = plt.colorbar(im3, ax=ax3, extend='both', shrink=0.8, ticks=[0, 0.5, 1, 1.5])
cbar3.set_label('Vr/Vs')
ax3.set_title('Rupture Speed')
ax3.set_xlabel('Depth (km)') 
ax3.set_ylabel('Along-strike distance (km)') 

# Final slip
ax4 = plt.subplot2grid((6, 1), (3, 0), colspan=1, rowspan=1)
vmin_value = np.nanmin(Slip_grid)
vmax_value = np.nanmax(Slip_grid)
im4 = ax4.imshow(Slip_grid.T, extent=fault_range, origin='lower', cmap='Reds', vmin=vmin_value, vmax=10, aspect='auto')
cs4 = ax4.contour(grid_x, grid_y, Slip_grid, ContourS, colors='k', linewidths=0.1)
cbar4 = plt.colorbar(im4, ax=ax4, extend='both', shrink=0.8, ticks=[0, 5, 10])
cbar4.set_label('Slip (m)')
ax4.set_title('Final Slip')
ax4.set_xlabel('Depth (km)') 
ax4.set_ylabel('Along-strike distance (km)') 

# Stress drop at strike direction
ax5 = plt.subplot2grid((6, 1), (4, 0), colspan=1, rowspan=1)
im5 = ax5.imshow(str_grid.T, extent=fault_range, origin='lower', cmap='RdBu', vmin=-30, vmax=30, aspect='auto')
cbar5 = plt.colorbar(im5, ax=ax5, extend='both', shrink=0.8, ticks=[-30, -15, 0, 15, 30])
cbar5.set_label('Stress Drop (MPa)')
ax5.set_title('Strike Stress Drop')
ax5.set_xlabel('Depth (km)') 
ax5.set_ylabel('Along-strike distance (km)') 

# Stress drop at dip direction
ax6 = plt.subplot2grid((6, 1), (5, 0), colspan=1, rowspan=1)
im6 = ax6.imshow(dip_grid.T, extent=fault_range, origin='lower', cmap='viridis', vmin=-30, vmax=30, aspect='auto')
cbar6 = plt.colorbar(im6, ax=ax6, extend='both', shrink=0.8, ticks=[-30, -15, 0, 15, 30])
cbar6.set_label('Stress Drop (MPa)')
ax6.set_title('Dip Stress Drop')
ax6.set_xlabel('Depth (km)') 
ax6.set_ylabel('Along-strike distance (km)') 

plt.tight_layout() 

plt.savefig("ps/"+Model_name+"-results.pdf",format="pdf")
print("pdf file is saved.")


