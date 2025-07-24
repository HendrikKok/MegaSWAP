#!/usr/bin/env python
import netCDF4 as nc
import numpy as np
import sys
import matplotlib.pyplot as plt
 
path = sys.argv[1].strip()
ds = nc.Dataset(path)
qmodf = ds.variables['qmodf']
qrun = ds.variables['qrun']
sc1 = ds.variables['sc1']
phead = ds.variables['phead']
qrain = ds.variables['qrain']
peva = ds.variables['peva']
reva = ds.variables['reva']
qbot = ds.variables['qbot']
time = ds.variables['time']

figure, ax = plt.subplot_mosaic(
    """
    24
    67
    """
)
ax["4"].plot(time[:],qmodf[:], '-')
ax["2"].plot(time[:],sc1[:], '-')
ax["6"].plot(time[:],phead[:], '-')
ax["7"].plot(time[:],qbot[:], '-')

ax["4"].set_title("qmodf")
ax["2"].set_title("sc1")
ax["6"].set_title("phead")
ax["7"].set_title("qrun")

plt.tight_layout()
plt.savefig("exchange_vars_coupled_combined.png")
plt.show()
plt.close()
    

figure2, ax2 = plt.subplot_mosaic(
    """
    8
    """
    )

ax2["8"].plot(phead[:,0], 'r-', label='box 1')
ax2["8"].plot(phead[:,1], 'b-', label='box 2')
ax2["8"].plot(phead[:,2], 'g-', label='box 3')
ax2["8"].plot(phead[:,3], 'm-', label='box 4')

ax2["8"].legend()
plt.tight_layout()
plt.savefig("msw_heads.png")
plt.show()
plt.close()

figure3, ax3 = plt.subplot_mosaic(
    """
    ab
    cd
    """
    )

ax3["a"].plot(time[:],qmodf[:], '-')
ax3["b"].plot(time[:],qrun[:], '-')
ax3["c"].plot(time[:],reva[:], 'r-',label='reva')
ax3["c"].plot(time[:],peva[:], 'm--',label='peva')
ax3["c"].plot(time[:],qrain[:], 'b--',label='qrain')
ax3["d"].plot(time[:],qbot[:], '-')

ax3["a"].set_title("qmodf")
ax3["b"].set_title("qrun")
ax3["c"].set_title("reva, peva, qrain")
ax3["c"].legend()
ax3["d"].set_title("qbot")

plt.tight_layout()
plt.savefig("msw_sat.png")
plt.show()
plt.close()
