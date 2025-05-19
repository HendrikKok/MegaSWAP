import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import imod
from src.mf6_simulation import run_experimental_non_coupled_model, run_non_coupled_model


# grb_file = r'c:\werkmap\coupler_model\mf6_model\model.dis.grb'
# hds_file = r'c:\werkmap\coupler_model\mf6_model\flow.hds'
# 
# 
# heads = imod.mf6.open_hds(grb_path=grb_file,hds_path=hds_file)
# 
# h1 = heads.isel(layer = 0, y=0, x=0).compute()
# 
# 
# plt.plot(h1)
# 
# 
# # mk chd-file
# 
# init_gwl = -3.0
# increment = 0.02
# 
# ntime = int(((0-init_gwl) / increment))
# gwl = init_gwl
# with open("chd.chd", "w") as f:
#     f.write('BEGIN OPTIONS \n')  
#     f.write('    SAVE_FLOWS\n')
#     f.write('END OPTIONS\n')  
#     f.write('\n')
#     f.write('BEGIN DIMENSIONS\n')
#     f.write('   MAXBOUND 1\n')
#     f.write('END DIMENSIONS\n')
#     f.write('\n')
#     for itime in range(1, ntime+1):
#         gwl+=increment
#         f.write(f'BEGIN PERIOD {itime}\n')
#         f.write(f'    1 1 1 {gwl}\n')
#         f.write(f'END PERIOD {itime}\n')
# 
# pass

















# msw inputs
qrch = np.array([0.0045] * 100 + [0.000] * 400)   # [0.0045] * 60 + [0.0026] * 20 + [0.0032] * 20 + [0.000] * 400
qpet = np.ones_like(qrch) * 0.0  # * 0.001
# qrch = np.array([0.0026]*30 + [0.0]*10 + [0.0032]*30 + [0.0026]*50)

msw_parameters = {
    "databse_path": r"c:\src\MegaSWAP\database\unsa_079.nc",
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "qpet": qpet,
    "surface_elevation": 0.0,
    "initial_gwl": -3.0,
    "initial_phead": -1.513561,  # -(0 - -6.9)
    "dtgw": 1.0,
    "area": 100.0,
    "max_infiltration": 0.0045, # 0.0036  
}
ntime = 200
megaswap, log = run_experimental_non_coupled_model(ntime, msw_parameters, 0.02)

figure, ax = plt.subplots(2)
ax[0].plot(log.ig)
ax[1].plot(log.fig)
plt.savefig('ig.png')
plt.close()


figure, ax = plt.subplots(2)
ax[0].plot(log.ip)
ax[1].plot(log.fip)
plt.savefig('ip.png')
plt.close()


tabel = xr.open_dataset(msw_parameters["databse_path"])
ig = np.arange(tabel.nxlig - 1, tabel.nuig + 1, 1)
ip = np.arange(tabel.nlip, tabel.nuip + 1, 1)
ib = np.arange(0, 18, 1)
svtb = tabel["svtb"].assign_coords(
    {
        "ip": ip,
        "ig": ig,
        "ib": ib,
    }
).fillna(0.0).compute()
qmrtb = tabel["qmrtb"].assign_coords(
    {
        "ip": ip,
        "ig": ig,
    }
).fillna(0.0).compute()


out = xr.DataArray(
    data = qmrtb.to_numpy(),
    coords = {
        'y': qmrtb.ip.to_numpy(),
        'x': qmrtb.ig.to_numpy(),
    },
    dims = ['y','x'],
)
imod.idf.save('qmrtb.idf', out)

svtb.isel(ib=0, drop=True).plot.surface()
plt.savefig('svtb.png')
plt.close()

figure, ax = plt.subplots(2)
for ig in svtb.ig:
    ax[0].plot(svtb.sel(ib=0, ig=ig), label = f"ig{ig.item()}")
for ip in svtb.ip:
    ax[1].plot(svtb.sel(ib=0, ip=ip), label = f"ip{ig.item()}")

ax[0].legend()
ax[1].legend()
plt.savefig('svtb_sliced.png')
plt.close()

sigma = svtb - qmrtb
figure, ax = plt.subplots(2)
for ig in svtb.ig:
    ax[0].plot(sigma.sel(ib=0, ig=ig), label = f"ig{ig.item()}")
for ip in svtb.ip:
    ax[1].plot(sigma.sel(ib=0, ip=ip), label = f"ip{ig.item()}")

ax[0].legend()
ax[1].legend()
plt.savefig('sigma_sliced.png')
plt.close()



figure, ax = plt.subplots(1)
c = np.arange(log.ig.size)
# c[0:100] = 0
(svtb.isel(ib=0, drop=True) - qmrtb).plot(ax=ax)
ax.scatter(log.ig+ log.fig, log.ip + log.fip ,c=c)
ax.set_xlabel('ig')
ax.set_ylabel('ip')
plt.savefig('ip_ig_scatter.png')
plt.close()

(svtb.isel(ib=0, drop=True) - qmrtb).plot.surface()
plt.savefig('sigma_tb.png')
plt.close()


(qmrtb).plot.surface()
plt.savefig('qmrtb.png')
plt.close()

pass