import numpy as np
from src.mf6_simulation import run_coupled_model, run_experimental_coupled_model
from scripts.plot_coupled_results import plot_results, plot_combined_results
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd


# msw inputs
# qrch = np.array([0.002] + [0.00] *119)   # 0.36
qrch = np.array([0.0026]*30 + [0.0]*10 + [0.0032]*30 + [0.0026]*250)
qpet = np.ones_like(qrch) * 0.0

msw_parameters = {
    "databse_path": r"database\unsa_079_100.nc",
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "qpet": qpet,
    "surface_elevation": 0.0,
    "initial_gwl": -3.0,
    "initial_phead": -1.513561,  # -(0 - -6.9)
    "dtgw": 1.0,
    "area": 100.0,
    "max_infiltration": 0.004, # 0.0036 
    "factor_ponding": 1.0,
}

# mf6 inputs
mf6_parameters = {
    "workdir": r"c:\src\MegaSWAP\mf6_model",
    "model_name": "model",
}

# chek s_mf6
ntime = 240 # 120  300
megaswap, log = run_experimental_coupled_model(ntime, mf6_parameters, msw_parameters)
# megaswap, log = run_coupled_model(ntime, mf6_parameters, msw_parameters)

time_min = 180


svat_per = pd.read_csv(r"c:\werkmap\coupler_model\metaswap\msw\csv\svat_per_0000000001.csv")
s1_msw = svat_per["      decS01(mm)"][time_min:ntime].to_numpy()
s2_msw = svat_per["      decS02(mm)"][time_min:ntime].to_numpy()
s3_msw = svat_per["      decS03(mm)"][time_min:ntime].to_numpy()
s = s1_msw + s2_msw + s3_msw


plt.plot(log.ds[time_min:ntime,-1], label='ds')
plt.plot(s, '--',label='ds_msw')
plt.savefig('ds.png')
plt.close()

figure, ax = plt.subplot_mosaic(
    """
    ab
    cd
"""
)
ax['a'].plot(log.ig[time_min:ntime], label='ig')
ax['c'].plot(log.fig[time_min:ntime], label='fig')
ax['b'].plot(log.ip[time_min:ntime,0:2], label='ip')
ax['d'].plot(log.fip[time_min:ntime,0:2], label='fip')
ax['a'].legend()
ax['b'].legend()
ax['c'].legend()
ax['d'].legend()
plt.savefig('index.png')
plt.close()



for ibox in range(4):
    plt.plot(log.sigma[time_min:ntime, ibox], label=f'ibox{ibox}')
plt.legend()
plt.savefig('sigma.png')
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



for ibox in range(0, 4):
    active= log.active[time_min:ntime,ibox]==1
    ig_slice = slice(log.ig[time_min:ntime].min(), log.ig[time_min:ntime].max())
    ip_slice = slice(log.ip[time_min:ntime,ibox].min(), log.ip[time_min:ntime,ibox].max())
    figure, ax = plt.subplots(1)
    c = np.arange(log.ig.size)
    (svtb.isel(ib=ibox, drop=True) - qmrtb).sel(ig=ig_slice, ip=ip_slice, drop=True).plot(ax=ax)
    ax.scatter(log.ig[time_min:ntime][active]+ log.fig[time_min:ntime][active], log.ip[time_min:ntime,ibox][active] + log.fip[time_min:ntime,ibox][active] ,c=c[time_min:ntime][active])
    ax.set_xlabel('ig')
    ax.set_ylabel('ip')
    plt.savefig(f'thetatb_scatter_box{ibox}.png')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    (svtb.isel(ib=ibox, drop=True) - qmrtb).sel(ig=ig_slice, ip=ip_slice, drop=True).plot.surface(ax=ax)
    ax.view_init(azim=150)
    plt.savefig(f'theta_surface_box{ibox}.png')
    plt.close()

    for ip_loc in range(int(log.ip[time_min:ntime,ibox].min().item()), int(log.ip[time_min:ntime,ibox].max().item()+1)):
        theta = (svtb.isel(ib=ibox, drop=True) - qmrtb).sel(ig=ig_slice, ip=ip_loc, drop=True)
        plt.plot(theta.ig,theta, label =f'ip={ip_loc}')
    plt.legend()
    plt.savefig(f'theta_slice_box{ibox}.png')
    plt.close()

for ibox in range(0, 4):
    active= log.active[time_min:ntime,ibox]==1
    ig_slice = slice(log.ig[time_min:ntime].min(), log.ig[time_min:ntime].max())
    ip_slice = slice(log.ip[time_min:ntime,ibox].min(), log.ip[time_min:ntime,ibox].max())
    figure, ax = plt.subplots(1)
    c = np.arange(log.ig.size)
    (svtb.isel(ib=ibox, drop=True)).sel(ig=ig_slice, ip=ip_slice, drop=True).plot(ax=ax)
    ax.scatter(log.ig[time_min:ntime][active]+ log.fig[time_min:ntime][active], log.ip[time_min:ntime,ibox][active] + log.fip[time_min:ntime,ibox][active] ,c=c[time_min:ntime][active])
    ax.set_xlabel('ig')
    ax.set_ylabel('ip')
    plt.savefig(f'svtb_scatter_box{ibox}.png')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    (svtb.isel(ib=ibox, drop=True)).sel(ig=ig_slice, ip=ip_slice, drop=True).plot.surface(ax=ax)
    ax.view_init(azim=150)
    plt.savefig(f'svtb_surface_box{ibox}.png')
    plt.close()


plot_results(log, megaswap, ntime, max_infiltration = msw_parameters["max_infiltration"],ntime_min = time_min)
plot_combined_results(log, megaswap, r'c:\werkmap\coupler_model', ntime, ntime_min = time_min)