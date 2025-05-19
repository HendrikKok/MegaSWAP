import numpy as np
from src.mf6_simulation import run_experimental_non_coupled_model, run_non_coupled_model
from scripts.plot_coupled_results import plot_results, plot_combined_results
import matplotlib.pyplot as plt
import pandas as pd
import imod
# read msw run

msw = pd.read_csv(r'c:\werkmap\coupler_model\metaswap_a\msw\csv\svat_per_0000000001.csv')

msw['       phrz01(m)']

# read mf6 run
grb_file = r'c:\werkmap\coupler_model\mf6_model\model.dis.grb'
hds_file = r'c:\werkmap\coupler_model\mf6_model\flow.hds'


heads = imod.mf6.open_hds(grb_path=grb_file,hds_path=hds_file)

h1 = heads.isel(layer = 0, y=0, x=0).to_numpy()


# msw inputs
qrch = np.array([0.0045] * 100 + [0.000] * 400)   # [0.0045] * 60 + [0.0026] * 20 + [0.0032] * 20 + [0.000] * 400
qpet = np.ones_like(qrch) * 0.0  # * 0.001
# qrch = np.array([0.0026]*30 + [0.0]*10 + [0.0032]*30 + [0.0026]*50)

msw_parameters = {
    "databse_path": r"n:\Deltabox\Postbox\Kok, Hendrik\vanRobert\unsa_079_020.nc",
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "qpet": qpet,
    "surface_elevation": 0.0,
    "initial_gwl": -2.98,
    "initial_phead": -1.513561,  # -(0 - -6.9)
    "dtgw": 1.0,
    "area": 100.0,
    "max_infiltration": 0.0045, # 0.0036  
}

# mf6 inputs
mf6_parameters = {
    "workdir": r"c:\src\MegaSWAP\mf6_model",
    "model_name": "model",
}

# chek s_mf6
ntime = 200 # 120
d1 = 0.02
d2 = 0.02
# megaswap, log = run_coupled_model(ntime, mf6_parameters, msw_parameters)
megaswap, log10 = run_experimental_non_coupled_model(ntime, msw_parameters, d1)
# megaswap, log20 = run_non_coupled_model(ntime, msw_parameters, d2)

# log10.phead[log10.phead < -1.4] = np.nan
# log20.phead[log20.phead < -1.4] = np.nan

fig ,ax = plt.subplots(1)
ax.plot(log10.phead[:,0], label = f'phead b1 d={d1*100} cm', color = 'green',linewidth=1)
ax.plot(log10.phead[:,1], label = f'phead b2 d={d1*100} cm', color = 'orange',linewidth=1)
ax.plot(log10.phead[:,2], label = f'phead b3 d={d1*100} cm', color = 'blue',linewidth=1)
ax.plot(log10.phead[:,3], label = f'phead b4 d={d1*100} cm', color = 'yellow',linewidth=1)
ax.plot(log10.phead[:,4], label = f'phead b5 d={d1*100} cm', color = 'purple',linewidth=1)


# plt.plot(log20.phead[:,0], '--' , label = f'box 1 d={d2*100} cm', color = 'green',linewidth=1)
# plt.plot(log20.phead[:,1], '--' , label = f'box 2 d={d2*100} cm', color = 'orange',linewidth=1)
# plt.plot(log20.phead[:,2], '--', label = f'box 3 d={d2*100} cm' , color = 'blue',linewidth=1)

ax.plot(msw['       phrz01(m)'], '--' ,label = f'phead b1 d={d2*100} cm', color = 'green',linewidth=1)
ax.plot(msw['       phrz02(m)'], '--' ,label = f'phead b2 d={d2*100} cm', color = 'orange',linewidth=1)
ax.plot(msw['       phrz03(m)'], '--' ,label = f'phead b3 d={d2*100} cm', color = 'blue',linewidth=1)
ax.plot(msw['       phrz04(m)'], '--' ,label = f'phead b4 d={d2*100} cm', color = 'yellow',linewidth=1)
ax.plot(msw['       phrz05(m)'], '--' ,label = f'phead b5 d={d2*100} cm', color = 'purple',linewidth=1)


#ax[1].plot(log10.mf6_head[:,1], label = f'heads prototype', color = 'green', linewidth=1)
#ax[1].plot(h1,'--', label = f'heads msw', color = 'orange', linewidth=1)



ax.legend()
plt.tight_layout()
plt.savefig(r"c:\src\MegaSWAP\results\pheads_diff.png")
plt.close()

plot_results(log10, megaswap, ntime,'exp')
# plot_results(log20, megaswap, ntime,'tra')
# plot_combined_results(log, megaswap, r'c:\werkmap\coupler_model', ntime)
