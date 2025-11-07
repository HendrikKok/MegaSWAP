import numpy as np
from src.mf6_simulation import run_experimental_non_coupled_model
from scripts.plot_wbal import plot_wbal
from scripts.plot_coupled_results import plot_results
import matplotlib.pyplot as plt


qrch = np.array([0.0045] * 60 + [0.0026] * 20 + [0.0032] * 20 + [0.000] * 400)
qpet = np.ones_like(qrch) * 0.001
gwl = -6
msw_parameters = {
    "databse_path": r"c:\src\MegaSWAP\database\unsa_079_100.nc",
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "qpet": qpet,
    "surface_elevation": 0.0,
    "initial_gwl": gwl,
    "initial_phead": -(0-gwl),
    "dtgw": 1.0,
    "area": 100.0,
    "max_infiltration": 0.0036, # 0.0045  
    "factor_ponding": 0.5,
}

ntime = 200
d1 = 0.02 # head change per delt
megaswap, log = run_experimental_non_coupled_model(ntime, msw_parameters, d1)

plot_wbal(ntime, log, qrch, r"c:\src\MegaSWAP\results")
plot_results(log, megaswap, ntime, max_infiltration = msw_parameters["max_infiltration"])


