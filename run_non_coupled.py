import numpy as np
from src.mf6_simulation import run_experimental_non_coupled_model
from scripts.plot_coupled_results import plot_results, plot_combined_results

# msw inputs
qrch = np.array([0.0045] * 60 + [0.0026] * 20 + [0.0032] * 20+ [0.000] * 100)   # 0.36
qpet = np.ones_like(qrch) * 0.001
# qrch = np.array([0.0026]*30 + [0.0]*10 + [0.0032]*30 + [0.0026]*50)

msw_parameters = {
    "databse_path": r"c:\werkmap\coupler_model\metaswap_a\unsa_001.nc",
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "qpet": qpet,
    "surface_elevation": 0.0,
    "initial_gwl": -1.0,
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
ntime = 200 # 120
# megaswap, log = run_coupled_model(ntime, mf6_parameters, msw_parameters)
megaswap, log = run_experimental_non_coupled_model(ntime, msw_parameters)

plot_results(log, megaswap, ntime)
# plot_combined_results(log, megaswap, r'c:\werkmap\coupler_model', ntime)
