#!/usr/bin/env python
import netCDF4 as nc
import numpy as np
import sys
import matplotlib.pyplot as plt
 
path = sys.argv[1].strip()
ds = nc.Dataset(path)
qmodf = ds.variables['qmodf']
ponding_stage = ds.variables['pond']
qrun = ds.variables['qrun']
sc1 = ds.variables['sc1']
gwl = ds.variables['gwl']
phead = ds.variables['phead']
theta = ds.variables['theta']
qrain = ds.variables['qrain']
peva = ds.variables['peva']
reva = ds.variables['reva']
reva_ponding = ds.variables['ponding_reva']
qbot = ds.variables['qbot']
qrot = ds.variables['qrot']
qrch = ds.variables['qrch']
nbox = ds.variables['nbox']
time = ds.variables['time']

settings = ds.variables['settings']
atts = ({k: settings.getncattr(k) for k in settings.ncattrs()})
zmax_ponding = atts['zmax_ponding']

sc1_list = []
qmodf_list = []
qrun_list = []
qsim_list = []
vpond_list = []
evpond_list = []
evsoil_list = []

sref1 = "D:/leander/MegaSWAP/git/MegaSWAP_standalone/MegaSWAP/results.txt"
with open(sref1,"r") as fref1:
    svals = fref1.readlines()
for sval in svals:
    values=[float(valuestr) for valuestr in sval.split()]
    sc1_list.append(values[0])
    qmodf_list.append(values[1])
    qrun_list.append(values[2])
    qsim_list.append(values[3])
    vpond_list.append(values[4])
    evpond_list.append(values[5])
    evsoil_list.append(values[6])
sc1ref = np.array(sc1_list)
qmodfref = np.array(qmodf_list)
qrunref = np.array(qrun_list)
qsimref = np.array(qsim_list)
vpondref = np.array(vpond_list)
evpondref = np.array(evpond_list)
evsoilref = np.array(evsoil_list)
mref = np.array(qsim_list)


figure2, ax2 = plt.subplot_mosaic(
    """
    8
    """,
figsize=(8, 8)
)

# theta versus phead
mask = (nbox[:]>=5)
ax2["8"].plot(theta[mask,4], phead[mask,4], 'c.', label='box 5')
mask = (nbox[:]>=4)
ax2["8"].plot(theta[mask,3], phead[mask,3], 'm.', label='box 4')
mask = (nbox[:]>=3)
ax2["8"].plot(theta[mask,2], phead[mask,2], 'g.', label='box 3')
mask = (nbox[:]>=2)
ax2["8"].plot(theta[mask,1], phead[mask,1], 'b.', label='box 2')
mask = (nbox[:]>=1)
ax2["8"].plot(theta[mask,0], phead[mask,0], 'r.', label='box 1')
box = ax2["8"].get_position()
ax2["8"].legend(loc='upper left', bbox_to_anchor=(1.2, 1))
ax2["8"].invert_yaxis()

plt.tight_layout()
plt.savefig("msw_theta_phead.png")
plt.show()
plt.close()
