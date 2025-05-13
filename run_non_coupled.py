import numpy as np
from src.mf6_simulation import run_experimental_non_coupled_model
from scripts.plot_coupled_results import plot_results, plot_combined_results
import matplotlib.pyplot as plt


# msw inputs
qrch = np.array([0.0045] * 60 + [0.0026] * 20 + [0.0032] * 20+ [0.000] * 400)   # 0.36
qpet = np.ones_like(qrch) * 0.001
# qrch = np.array([0.0026]*30 + [0.0]*10 + [0.0032]*30 + [0.0026]*50)

msw_parameters = {
    "databse_path": r"c:\werkmap\coupler_model\metaswap_a\unsa_001.nc",
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "qpet": qpet,
    "surface_elevation": 0.0,
    "initial_gwl": -3.0,
    "initial_phead": -1.513561,  # -(0 - -6.9)
    "dtgw": 1.0,
    "area": 100.0,
    "max_infiltration": 0.0032, # 0.0036  
}

# mf6 inputs
mf6_parameters = {
    "workdir": r"c:\src\MegaSWAP\mf6_model",
    "model_name": "model",
}

# chek s_mf6
ntime = 500 # 120
d1 =-0.005
d2 = -0.01
# megaswap, log = run_coupled_model(ntime, mf6_parameters, msw_parameters)
megaswap, log10 = run_experimental_non_coupled_model(ntime, msw_parameters, d1)
megaswap, log20 = run_experimental_non_coupled_model(ntime, msw_parameters, d2)

log10.phead[log10.phead < -1.4] = np.nan
log20.phead[log20.phead < -1.4] = np.nan

plt.plot(log10.phead[:,0], label = f'box 1 d={d1*100} cm', color = 'green')
plt.plot(log10.phead[:,1], label = f'box 2 d={d1*100} cm', color = 'orange')
plt.plot(log10.phead[:,2], label = f'box 3 d={d1*100} cm', color = 'blue')

plt.plot(log20.phead[:,0], '--' , label = f'box 1 d={d2*100} cm', color = 'green')
plt.plot(log20.phead[:,1], '--' , label = f'box 2 d={d2*100} cm', color = 'orange')
plt.plot(log20.phead[:,2], '--', label = f'box 3 d={d2*100} cm' , color = 'blue')
plt.legend()
plt.tight_layout()
plt.savefig(r"c:\src\MegaSWAP\results\pheads_diff.png")
plt.close()





# plot_results(log, megaswap, ntime)
# plot_combined_results(log, megaswap, r'c:\werkmap\coupler_model', ntime)
