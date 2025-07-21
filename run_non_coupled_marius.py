import numpy as np
from src.mf6_simulation import NonCoupledExperimentalSimulation
from scripts.plot_coupled_results import plot_results
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
# read msw run

results_path = 'D:\\leander\\GWSobek\\GWSobek\\branches\\sss_prototype\\src\\voorMarius\\ExampleSSS_coupled_Marius\\'
results_file = 'results_save.nc'
results_file = 'results_.nc'
dff = nc.Dataset(results_path+results_file, 'r')
settings = dff.variables['settings']
atts = ({k: settings.getncattr(k) for k in settings.ncattrs()})


# ground water level
h1 = dff.variables["gwl"][:] 

# parameters and rainall and evap from the fortran netcdf file
qrain = dff.variables["qrain"][:]
qpet = dff.variables["peva"][:]
# input("press to continue")  # press_a_key_to_continue


qrot = dff.variables["qrot"][:]

msw_parameters = {
    "databse_path": atts['unsa_db'],
    "rootzone_dikte": atts['dprz'],
    "qrch": qrain,
    "qpet": qpet,
    "qrot": qrot,
    "gwl": h1,
    "surface_elevation": atts["top"],
    "initial_gwl": atts["init_gwl"],
    "initial_phead": atts["init_phead"],
    "dtgw": atts["dtgw"],
    "area": atts["area"],
    "max_infiltration": atts["maxinf"],
}

# run simulation

ntime = np.size(h1)
sim = NonCoupledExperimentalSimulation(msw_parameters)
sim.run(ntime)

# evaluate results
# actual evaporation
fig, ax = plt.subplots(1)
evap_f = dff.variables["reva"][:]
ax.plot(evap_f[:], label = f'evap_fortran', color = 'green', linewidth=0.5, marker='o')
ax.plot(sim.log.evap_soil, label = f'evap_soil', color = 'black', linewidth=1)
plt.show()

# recharge to soil
fig, ax = plt.subplots(1)
qrch_f = dff.variables["qrch"][:]
ax.plot(qrch_f[:], label = f'qrch_fortran', color = 'green', linewidth=0.5, marker='o')
ax.plot(sim.log.qrch, label = f'recharge', color = 'black',linewidth=1)
plt.show()

# heads box 1
fig, ax = plt.subplots(1)
phead_f = dff.variables["phead"][:]
ax.plot(phead_f[:,0], label = f'phead b1_fortran', color = 'green', linewidth=0.5, marker='o')
ax.plot(sim.log.phead[:,0], label = f'phead b1', color = 'black',linewidth=1)
plt.show()

# heads all boxes
fig, ax = plt.subplots(1)
ax.plot(sim.log.phead[:,0], label = f'phead b1', color = 'green',linewidth=1)
ax.plot(sim.log.phead[:,1], label = f'phead b2', color = 'orange',linewidth=1)
ax.plot(sim.log.phead[:,2], label = f'phead b3', color = 'blue',linewidth=1)
ax.plot(sim.log.phead[:,3], label = f'phead b4', color = 'yellow',linewidth=1)
ax.plot(sim.log.phead[:,4], label = f'phead b5', color = 'purple',linewidth=1)

phead_f = dff.variables["phead"][:]
ax.plot(phead_f[:,0], label = f'phead b1_fortran', color = 'green', linewidth=0.5, marker='o')
ax.plot(phead_f[:,1], label = f'phead b2_fortran', color = 'orange', linewidth=0.5, marker='o')
ax.plot(phead_f[:,2], label = f'phead b3_fortran', color = 'blue', linewidth=0.5, marker='o')
ax.plot(phead_f[:,3], label = f'phead b4_fortran', color = 'yellow', linewidth=0.5, marker='o')
ax.plot(phead_f[:,4], label = f'phead b5_fortran', color = 'purple', linewidth=0.5, marker='o')

ax.plot(sim.log.phead[:,0], label = f'phead b1', color = 'black',linewidth=1)
ax.plot(sim.log.phead[:,1], label = f'phead b2', color = 'black',linewidth=1)
ax.plot(sim.log.phead[:,2], label = f'phead b3', color = 'black',linewidth=1)
ax.plot(sim.log.phead[:,3], label = f'phead b4', color = 'black',linewidth=1)
ax.plot(sim.log.phead[:,4], label = f'phead b5', color = 'black',linewidth=1)

plt.show()
