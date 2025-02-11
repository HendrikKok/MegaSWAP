import numpy as np
from src.mf6_simulation import run_coupled_model
from scripts.plot_coupled_results import plot_results, plot_combined_results

# msw inputs
qrch = np.array([0.0026] * 160)
msw_parameters = {
    "databse_path": r"d:\werkmap\prototype_metaswap\coupler_model\metaswap_a\unsa_001.nc",
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "surface_elevation": 0.0,
    "initial_gwl": -3.0,
    "initial_phead": -1.513561,  # -(0 - -6.9)
    "dtgw": 1.0,
}

# mf6 inputs
mf6_parameters = {
    "workdir": r"d:\werkmap\prototype_metaswap\MegaSWAP\mf6_model",
    "model_name": "model",
}

# chek s_mf6

ntime = 70
megaswap, log = run_coupled_model(ntime, mf6_parameters, msw_parameters)

plot_results(log, megaswap, ntime)
plot_combined_results(log, megaswap, r'd:\werkmap\prototype_metaswap\coupler_model', ntime)
