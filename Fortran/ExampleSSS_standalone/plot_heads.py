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
nbox = ds.variables['nbox']
qbot = ds.variables['qbot']
qrot = ds.variables['qrot']
qrch = ds.variables['qrch']
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
    2
    3
    6
    4
    8
    """,
figsize=(8, 8)
)

# heads
ax2["8"].plot(phead[:,4], 'c-', label='box 5')
ax2["8"].plot(phead[:,3], 'm-', label='box 4')
ax2["8"].plot(phead[:,2], 'g-', label='box 3')
ax2["8"].plot(phead[:,1], 'b-', label='box 2')
ax2["8"].plot(phead[:,0], 'r-', label='box 1')
box = ax2["8"].get_position()
ax2["8"].legend(loc='upper left', bbox_to_anchor=(1.2, 1))

# recharge, qrot, qsim
ax2["4"].plot(qrch[:], 'g-', label='qrch')
ax2["4"].plot(qrot[:], 'r-', label='qrot')
ax2["4"].plot(qbot[:], 'c-', label='qsim')
ax2["4"].legend(loc='upper left', bbox_to_anchor=(1.2, 1))

# ground water level
ax2["2"].plot(gwl[:]/100.0, 'b-', label='groundwater level')
ax2["2"].legend(loc='upper left', bbox_to_anchor=(1.2, 1))
ax22=ax2["2"].twinx()
print(nbox[:])
ax22.plot(nbox[:], 'c-', label='deepest box')

# ponding level
ax2["3"].plot(ponding_stage[:], 'C1-', label='ponding level')
ax2["3"].plot(
    np.array([1,np.size(ponding_stage)]),
    np.array([zmax_ponding,zmax_ponding]), 
   'k--', label='max level')
ax2["3"].legend(loc='upper left', bbox_to_anchor=(1.2, 1))

# top systeem, verdamping, neerslag
ax2["6"].plot(qrain[:], 'g-', label='rain')
ax2["6"].plot(peva[:], 'r-', label='evpot')
ax2["6"].plot(reva[:], 'm-', label='evact')
ax2["6"].legend(loc='upper left', bbox_to_anchor=(1.2, 1))

for key, val in ax2.items():
    box = val.get_position()
    val.set_position([box.x0, box.y0, box.width * 1.1, box.height])

# open referentie voor head in box 0 
h0_list = []
h1_list = []
h2_list = []
h3_list = []
sref0 = "D:/leander/MegaSWAP/git/MegaSWAP_standalone/MegaSWAP/head_0.txt"
with open(sref0,"r") as fref0:
    svals = fref0.readlines()
for sval in svals:
    values=[float(valuestr) for valuestr in sval.split()]
    h0_list.append(values[0])
    h1_list.append(values[1])
    h2_list.append(values[2])
    h3_list.append(values[3])




h0ref = np.array(h0_list)
h1ref = np.array(h1_list)
h2ref = np.array(h2_list)
h3ref = np.array(h3_list)

plt.tight_layout()
plt.savefig("msw_heads.png")
plt.show()
plt.close()
