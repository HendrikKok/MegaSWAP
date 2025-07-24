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
dsdt = ds.variables['ds_dt']
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
print(qmodfref)


figure, ax = plt.subplot_mosaic(
    """
    24
    67
    """
)
ntime = len(qmodf[:])
len = len(qmodf)
#ax["4"].plot(time[:],qmodfref[:ntime], 'k.-')
#ax["2"].plot(time[:],sc1ref[:ntime], 'k.-')
ax["6"].plot(time[:],phead[:ntime], 'k.-')
#ax["7"].plot(time[:],qsimref[:ntime], 'k.-')
ax["4"].plot(time[:],qmodf[:], 'r-')
ax["2"].plot(time[:],sc1[:], 'r-')
ax["6"].plot(time[:],phead[:], 'r-')
ax["7"].plot(time[:],qbot[:]/100., 'r-')

ax["4"].set_title("qmodf")
ax["2"].set_title("sc1")
ax["6"].set_title("phead")
ax["7"].set_title("qbot")


plt.tight_layout()
plt.savefig("exchange_vars_coupled_combined.png")
plt.show()
plt.close()



figure7, ax7 = plt.subplot_mosaic(
    """
    9
    """
    )

ax7["9"].plot(dsdt[:], 'm-', label='qrch-qrot-qsim')
ax77=ax7["9"].twinx()
plt.tight_layout()
plt.savefig("volumes.png")
plt.show()
plt.close()




figure5, ax5 = plt.subplot_mosaic(
    """
    8
    """
    )

ax5["8"].plot(theta[:,3], 'm-', label='box 4')
ax5["8"].plot(theta[:,2], 'g-', label='box 3')
ax5["8"].plot(theta[:,1], 'b-', label='box 2')
ax5["8"].plot(theta[:,0], 'r+', label='box 1')


ax55=ax5["8"].twinx()
ax55.plot(gwl[:], 'c-', label='gwl')
ax55.plot(gwl[:]*0.0, 'k-.')
ax55.legend()
plt.tight_layout()
plt.savefig("theta.png")
plt.show()
plt.close()



    

figure2, ax2 = plt.subplot_mosaic(
    """
    8
    """
    )

#ax2["8"].plot(phead[:,3], 'm+', label='box 4')
#ax2["8"].plot(phead[:,2], 'g+', label='box 3')
#ax2["8"].plot(phead[:,1], 'b+', label='box 2')
#ax2["8"].plot(phead[:,0], 'r+', label='box 1')

ax2["8"].plot(phead[:,3], 'm-', label='box 4')
ax2["8"].plot(phead[:,2], 'g-', label='box 3')
ax2["8"].plot(phead[:,1], 'b-', label='box 2')
ax2["8"].plot(phead[:,0], 'r-', label='box 1')


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
#ax2["8"].plot(h3ref, 'k--', label='ref')
#ax2["8"].plot(h2ref, 'k--', label='box 3, ref')
#ax2["8"].plot(h1ref, 'k--', label='box 2, ref')
#ax2["8"].plot(h0ref, 'k--', label='box 1, ref')

#ax2["8"].plot(h3ref, 'm-', label='box 4, ref')
#ax2["8"].plot(h2ref, 'g-', label='box 3, ref')
#ax2["8"].plot(h1ref, 'b-', label='box 2, ref')
#ax2["8"].plot(h0ref, 'r-', label='box 1, ref')

ax2["8"].legend()
plt.tight_layout()
plt.savefig("msw_heads.png")
plt.show()
plt.close()




#figure2a, ax2a = plt.subplot_mosaic(
#    """
#    a
#    """
#    )
#
#ax2a["a"].plot(phead[:,0], 'r+', label='box 1')
#ax2a["a"].plot(h0ref, 'k-', label='box 1, ref')
#plt.tight_layout()
#plt.savefig("msw_heads_0.png")
#plt.show()
#plt.close()
#
#
#figure2b, ax2b = plt.subplot_mosaic(
#    """
#    b
#    """
#    )
#ax2b["b"].plot(phead[:,1], 'r+', label='box 2')
#ax2b["b"].plot(h1ref, 'k-', label='box 2, ref')
#plt.tight_layout()
#plt.savefig("msw_heads_1.png")
#plt.show()
#plt.close()
#
#
#figure2c, ax2c = plt.subplot_mosaic(
#    """
#    c
#    """
#    )
#ax2c["c"].plot(phead[:,2], 'r+', label='box 3')
#ax2c["c"].plot(h2ref, 'k-', label='box 3, ref')
#plt.tight_layout()
#plt.savefig("msw_heads_2.png")
#plt.show()
#plt.close()
#
#
#figure2d, ax2d = plt.subplot_mosaic(
#    """
#    d
#    """
#    )
#ax2d["d"].plot(phead[:,3], 'r+', label='box 3')
#ax2d["d"].plot(h3ref, 'k-', label='box 3, ref')
#plt.tight_layout()
#plt.savefig("msw_heads_3.png")
#plt.show()
#plt.close()


figure3, ax3 = plt.subplot_mosaic(
    """
    ab
    cd
    """
    )

ax3["a"].plot(time[:],ponding_stage[:], '-')
ax3["b"].plot(time[:],qrun[:], '-')
ax3["c"].plot(time[:],reva[:], 'r-',label='reva')
ax3["c"].plot(time[:],reva_ponding[:], 'c-',label='reva_ponding')
ax3["c"].plot(time[:],peva[:], 'm--',label='peva')
ax3["c"].plot(time[:],qrain[:], 'b--',label='qrain')
ax3["d"].plot(time[:],qbot[:], '-')

ax3["a"].set_title("Ponding level")
ax3["b"].set_title("qrun")
ax3["c"].set_title("reva, peva, qrain")
ax3["c"].legend()
ax3["d"].set_title("qbot")

plt.tight_layout()
plt.savefig("msw_sat.png")
plt.show()
plt.close()
