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
time = ds.variables['time']

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

figure3, ax3 = plt.subplot_mosaic(
    """
    b
    a
    c
    """
    )

ax3["a"].plot(time[:],ponding_stage[:], '.-')
ax3["b"].plot(time[:],qrun[:], '-')
ax3["c"].plot(time[:],peva[:], 'k--',label='peva')
#ax3["c"].plot(time[:],reva[:], 'r-',label='reva')
#ax3["c"].plot(time[:],peva[:], 'm--',label='peva')
ax3["c"].step(time[:],reva[:], 'r-',label='reva')
ax3["c"].step(time[:],-reva_ponding[:], 'g-',label='reva_ponding')

ax3["a"].set_title("Ponding level")
ax3["b"].set_title("qrun")
ax3["c"].set_title("reva, peva, qrain")
ax3["c"].legend()

plt.tight_layout()
plt.savefig("msw_sat.png")
plt.show()
plt.close()
