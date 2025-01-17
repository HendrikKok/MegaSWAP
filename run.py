import numpy as np


from src.coupled_simulation import run_model
from scripts.plot_coupled_results import plot_results
        
# msw inputs
qrch = np.array([0.0026]*160)
msw_parameters = {
    "databse_path": 'database\\unsa_300.nc',
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "surface_elevation": 0.0,
    "initial_gwl": -3.0,
    "initial_phead": -1.5,  # -(0 - -6.9)
    "dtgw": 1.0,
}

# mf6 inputs
mf6_parameters = {
    "workdir": r"d:\werkmap\prototype_metaswap\MegaSWAP\mf6_model",
    "model_name": "model"
}

ntime = 110
megaswap, log  = run_model(ntime, mf6_parameters, msw_parameters)

plot_results(log, megaswap)