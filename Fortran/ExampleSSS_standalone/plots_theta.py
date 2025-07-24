#!/usr/bin/env python
import netCDF4 as nc
import xarray as xr

import numpy as np
import sys
import matplotlib.pyplot as plt
 
# open timeseries file and table files
ncpath = "D:/leander/MetaSWAP/test_compare_megaswap/van_hendrik_new/metaswap_a/unsa_nc/"
path = sys.argv[1].strip()   # path to the result time series
ds = xr.open_dataset(path)
unsadb_path = ncpath+'/'+ds%unsa_db
downscalingdb_path = ncpath+'/'+ds%downscaling_db
ds_unsadb = xr.open_dataset(unsadb_path)
ds_downscaling = xr.open_dataset(downscalingdb_path)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

nodenr = int(sys.argv[2]) 
boxnr = int(sys.argv[3]) 
theta2D = ds_downscaling.sel(inode=nodenr).variables['thetatb'][:].to_numpy()

theta = ds.sel(inode=nodenr).variables['theta'][:].to_numpy()
gamma = ds.sel(ib=boxnr).variables['gamma'][:].to_numpy()
phi = ds.variables['phi'][:].to_numpy()
time = ds.variables['time'][:].to_numpy()

ip = np.arange(np.shape(theta2D)[0])
ig = np.arange(np.shape(theta2D)[1])

X, Y = np.meshgrid(ig,ip)

# Plot the surface.
#surf = ax.plot_surface(X, Y, theta, cmap=cm.coolwarm,
#                      linewidth=0, antialiased=False, edgecolors='k')
surf = ax.plot_surface(X, Y, theta2D, cmap=cm.coolwarm,
                       linewidth=0.5, antialiased=False, edgecolors='k')
ax.set_xlabel('gamma')
ax.set_ylabel('phi')
ax.set_zlabel('theta')

# Plot the timesteps.

plt.show()

