import numpy as np
import matplotlib.pyplot as plt

from megaswap import MegaSwap

qrch = np.array([0.0026]*160)
parameters = {
    "databse_path": 'database\\unsa_300.nc',
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "surface_elevation": 0.0,
    "initial_gwl": -3.0,
    "initial_phead": -1.5,  # -(0 - -6.9)
    "dtgw": 1.0,
}

megaswap = MegaSwap(parameters)
megaswap.s_mf6 = 0.0 # dummy value

# stand alone MegaSwap run
ntime = 10
niter = 1
gwl = parameters['initial_gwl']

# logging 
phead_log = np.full((ntime, 18), np.nan)
gwl_log = np.full(ntime, np.nan)
nbox_log = np.full(ntime, np.nan)
megaswap.sc1 = 0.1
for itime in range(ntime):
    vsim = megaswap.prepare_timestep(itime)
    gwl = np.copy(megaswap.get_gwl())
    # sc1, nbox, _ = megaswap.do_iter(gwl, 0)
    # megaswap.qmodf = 0.0
    # megaswap.finalise_iter()
    # megaswap.storage_formulation.finalise()
    
    
    qmodf = megaswap.finalise_timestep(gwl)
    phead_log[itime,:] = np.copy(megaswap.phead)  # logging
    gwl_log[itime] = np.copy(gwl)

# plot pheads
max_box = 4
box_top = megaswap.database.box_top
box_bottom = megaswap.database.box_bottom
figure, ax = plt.subplot_mosaic(
    """
    00113
    22113
    """
) 

n = int(ntime/10)
if ntime < 10:
    n=1
colors = []
for ibox in range(max_box):
    ax['0'].plot(phead_log[:,ibox], label = f"h{ibox}")
ax['0'].legend()

for itime in range(0,ntime,n):
    p = np.repeat(phead_log[itime,0:max_box],2)
    y = np.stack([box_top[0:max_box],box_bottom[0:max_box]],axis=1).ravel()
    plot = ax['1'].plot(p, y, label = f"t={itime}")
    colors.append(plot[0].get_color())
pmin = phead_log[np.isfinite(phead_log)].min()
pmax = phead_log[np.isfinite(phead_log)].max()
ax['1'].hlines(0.0,pmin,pmax, color='grey')
for ibox in range(max_box):
    ax['1'].hlines(box_bottom[ibox],pmin,pmax, color='grey')

ax['2'].plot(nbox_log, label = 'active boxes')
ax['2'].legend()

ax['3'].hlines(0.0,0,1, color='grey')
for ibox in range(max_box):
    ax['3'].hlines(box_bottom[ibox],0,1, color='grey')
icol = 0
for itime in range(0,ntime,n):
    head = gwl_log[itime]
    ax['3'].hlines(head,0,1, color= colors[icol], label = f"t={itime}")
    icol+=1
ax['3'].legend()
plt.tight_layout()
plt.savefig("pheads_stand_alone.png")
plt.close()

figure, ax = plt.subplot_mosaic(
    """
    0
    """
) 
ax['0'].plot(gwl_log[:], label = 'msw-heads')
ax['0'].legend()
# ax['4'].legend()
plt.tight_layout()
plt.savefig("msw_heads.png")
plt.close()