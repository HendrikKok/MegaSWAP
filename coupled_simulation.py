import datetime
import os

import numpy as np
from xmipy import XmiWrapper
import matplotlib.pyplot as plt

from megaswap import MegaSwap

# Type hints
FloatArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray

class Logging:
    def __init__(self, ntime: int):
        self.phead = np.full((ntime,18), np.nan)
        self.nbox = np.full(ntime, np.nan)
        self.mf6_head = np.full(ntime, np.nan)
        self.msw_head = np.full(ntime, np.nan)
        self.vsim = np.full(ntime, np.nan)
        self.qmodf = np.full(ntime, np.nan)
        self.sc1 = np.full((ntime,2000), np.nan)
        self.sf_type = np.full((ntime,2000), np.nan)

class CoupledSimulation:
    """
    Run all stress periods in a simulation
    """

    def __init__(self, wdir: str, name: str, msw_parameters: dict):
        self.modelname = name
        self.mf6 = XmiWrapper(lib_path="libmf6.dll", working_directory=wdir)
        self.mf6.initialize()
        self.mf6_head = self.mf6.get_value_ptr("MODEL/X")
        self.mf6_sto = self.mf6.get_value_ptr("MODEL/STO/SS")
        self.mf6_rch = self.mf6.get_value_ptr("MODEL/RCH-1/RECHARGE")
        self.max_iter = self.mf6.get_value_ptr("SLN_1/MXITER")[0]
        self.msw = MegaSwap(msw_parameters)
        self.log = Logging(msw_parameters['qrch'].size)

        # (f"Initialized model with {self.ncell} cells")

    def do_iter(self, sol_id: int) -> bool:
        """Execute a single iteration"""
        has_converged = self.mf6.solve(sol_id)
        return has_converged

    def update(self, iperiod:int):
        # We cannot set the timestep (yet) in Modflow
        # -> set to the (dummy) value 0.0 for now        
        self.mf6.prepare_time_step(0.0)
        vsim = self.msw.prepare_timestep(iperiod)
        self.mf6_rch[:] = vsim
        
        self.mf6.prepare_solve(1)
        # Convergence loop
        for kiter in range(1, self.max_iter + 1):
            sc1, nbox, sf_type = self.msw.do_iter(self.mf6_head[0])
            self.mf6_sto[0] = sc1
            has_converged = self.do_iter(1)
            if has_converged and kiter > 5:
                break
            self.log.sc1[iperiod,kiter - 1] = sc1
            self.log.sf_type[iperiod,kiter - 1] = sf_type
        self.mf6.finalize_solve(1)
        self.log.msw_head[iperiod] = self.msw.get_gwl()
        self.msw.finalise_iter()
        
        # Finish timestep
        self.mf6.finalize_time_step()
        self.log.qmodf[iperiod] = self.msw.finalise_timestep(self.mf6_head[0])
        current_time = self.mf6.get_current_time()
        
        self.log.mf6_head[iperiod] = self.mf6_head[0]
        self.log.phead[iperiod,:] = self.msw.phead
        self.log.nbox[iperiod] = nbox
        self.log.vsim[iperiod] = vsim
        return current_time

    def get_times(self):
        """Return times"""
        return (
            self.mf6.get_start_time(),
            self.mf6.get_current_time(),
            self.mf6.get_end_time(),
        )

    def run(self, periods):
        iperiod = 0
        _, current_time, end_time = self.get_times()
        while (current_time < end_time) and iperiod < periods:
            # print(f"MF6 starting period {iperiod}")
            current_time = self.update(iperiod)
            iperiod += 1
        print(f"Simulation terminated normally for {periods} periods")

    def finalize(self):
        self.mf6.finalize()

def run_model(periods, msw_parameters: dict):
    wdir = r"d:\werkmap\prototype_metaswap\MegaSWAP\mf6_model"
    name = "model"
    sim = CoupledSimulation(wdir, name, msw_parameters)
    start = datetime.datetime.now()
    sim.run(periods)
    end = datetime.datetime.now()
    print(end - start)
    sim.finalize()
    return sim.msw, sim.log


# msw inputs
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

ntime = 120
megaswap, log  = run_model(ntime, parameters)

phead_log = log.phead
nbox_log = log.nbox
gwl_log = log.mf6_head
vsim_log = log.vsim
qmodf_log = log.qmodf
sc1_log = log.sc1

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
ax['1'].legend()
plt.tight_layout()
plt.savefig("pheads_coupled.png")
plt.close()


figure, ax = plt.subplot_mosaic(
    """
    01
    04
    23
    """
) 

n = int(ntime/10)
if ntime < 10:
    n=1
colors = []
for ibox in range(max_box):
    ax['0'].plot(phead_log[:,ibox], label = f"h{ibox}")
ax['0'].legend()

ax['1'].plot(log.vsim, label = 'vsim')
ax['1'].plot(log.qmodf, label = 'qmodf')
ax['1'].legend()

for ii in range(5):
    ax['2'].plot(log.sc1[:,ii], label = 'sc1')
    ax['4'].plot(log.sf_type[:,ii], label = 's-formulation')
ax['2'].legend()

ax['3'].plot(log.mf6_head, label = 'mf6-heads')
ax['3'].plot(log.msw_head, label = 'msw-heads')
ax['3'].legend()


# ax['4'].legend()


plt.tight_layout()
plt.savefig("exchange_vars_coupled.png")
plt.close()