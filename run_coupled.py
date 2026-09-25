import numpy as np
from src.mf6_simulation import run_coupled_model, run_experimental_coupled_model
from scripts.plot_coupled_results import plot_results, plot_combined_results
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scripts.plot_wbal import plot_wbal


# msw inputs
# qrch = np.array([0.002] + [0.00] *119)   # 0.36
qrch = np.array([0.0026]*30 + [0.0]*10 + [0.0032]*30 + [0.0026]*250)
# qrch = np.array([0.0046]*30 + [0.0]*10 + [0.0042]*30 + [0.0046]*250)
qpet = np.ones_like(qrch) * 0.0

msw_parameters = {
    "databse_path": r"c:\werkmap\coupler_model\metaswap\unsa_nc3\unsa_001_100.nc",  # 079     
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "qpet": qpet,
    "surface_elevation": 0.0,
    "initial_gwl": -3.0,
    "initial_phead": -1.5,  # -(0 - -6.9)
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

tabel = xr.open_dataset(msw_parameters["databse_path"]).load()
# tabel = tabel.where(tabel > 0.0).fillna(0.0)
for ibox, i in enumerate([112,113,114,115]):
    test = np.loadtxt(rf'c:\werkmap\coupler_model\metaswap\fort.{i}')
    
    figure = plt.figure(figsize=(10, 5))
    ax = [
        figure.add_subplot(1, 2, 1, projection='3d'),
        figure.add_subplot(1, 2, 2, projection='3d')
    ]
    nrow,ncol = test.shape
    test[test>0.0]=0.0
    xr.DataArray(
        data = test,
        coords = {
            'ip': np.arange(nrow),
            'ig': np.arange(ncol),
        },
        dims = ['ip','ig']

    ).plot.surface(ax=ax[0],cmap='jet')
    svtb = tabel['svtb'].isel(ib=ibox, drop=True)
    svtb = svtb.where(svtb < 0.0).fillna(0.0)
    svtb.plot.surface(ax=ax[1],cmap='jet')
    plt.savefig(f'svtb_combined_{ibox+1}.png')
    plt.close()

# figure = plt.figure(figsize=(10, 5))



# for i in range(4):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     tabel['svtb'].isel(ib=i, drop=True).plot.surface(ax=ax)
#     plt.savefig(f'svtb_sss_surface_box{i+1}.png')
#     plt.close()

np.savetxt('test.csv',tabel['svtb'].isel(ib=0).to_numpy(), fmt='%.5f')
igdc = pd.DataFrame(
    data={"index_gwtb": tabel.igdc.to_numpy().astype(dtype=np.int32)},
    index=np.arange(tabel.igdcmn, tabel.igdcmx + 1, 1),
)

plt.figure()
plt.plot(igdc["index_gwtb"])
# plt.ylim(0,300)
plt.savefig('igdc.png')
plt.close()


# chek s_mf6
ntime = 250 # 120  300
megaswap, log = run_experimental_coupled_model(ntime, mf6_parameters, msw_parameters)
# megaswap, log = run_coupled_model(ntime, mf6_parameters, msw_parameters)

time_min = 180 # 180

plot_wbal(ntime, log, qrch, r"c:\src\MegaSWAP\results")
plot_results(log, megaswap, ntime, max_infiltration = msw_parameters["max_infiltration"],ntime_min = time_min)
plot_combined_results(log, megaswap, r'c:\werkmap\coupler_model', ntime, ntime_min = time_min)

svat_per = pd.read_csv(r"c:\werkmap\coupler_model\metaswap\msw\csv\svat_per_0000000001.csv")
s1_msw = svat_per["      decS01(mm)"][time_min:ntime].to_numpy()
s2_msw = svat_per["      decS02(mm)"][time_min:ntime].to_numpy()
s3_msw = svat_per["      decS03(mm)"][time_min:ntime].to_numpy()
sc1_msw =svat_per["    sc1(m3/m2/m)"][time_min:ntime].to_numpy()
s = s1_msw + s2_msw + s3_msw

figure, ax = plt.subplots(2)
ax[0].plot(log.ds[time_min:ntime,-1], label='ds')
ax[1].plot(s/1000., '--',label='ds_msw')
ax[0].legend()
ax[1].legend()
plt.savefig('ds.png')
plt.close()


figure, ax = plt.subplot_mosaic(
    """
    a
    a
    a
    b
"""
)
ax['a'].plot(log.sc1[time_min:ntime,-1], label='sc1')
ax['a'].plot(log.sc2[time_min:ntime], '--',label='sc2')
ax['a'].plot(sc1_msw,label='sc_msw')
ax['b'].plot(log.active[time_min:ntime].sum(axis=1),label='nbox')


ax['a'].legend()
ax['a'].set_ylim(0,0.2)
plt.savefig('sc.png')
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
ig_slice = slice(8,30)
ip_slice = slice(0, 30)
for ibox in range(0, 4):
    active= log.active[time_min:ntime,ibox]==1
    # ig_slice = slice(log.ig[time_min:ntime].min(), log.ig[time_min:ntime].max())
    # ip_slice = slice(log.ip[time_min:ntime,ibox].min(), log.ip[time_min:ntime,ibox].max())
    figure, ax = plt.subplots(1)
    c = np.arange(log.ig.size)
    (svtb.isel(ib=ibox, drop=True) - qmrtb).sel(ig=ig_slice, ip=ip_slice, drop=True).plot(ax=ax)
    ax.scatter(log.ig[time_min:ntime][active]+ log.fig[time_min:ntime][active], log.ip[time_min:ntime,ibox][active] + log.fip[time_min:ntime,ibox][active] ,c=c[time_min:ntime][active],cmap='jet')
    ax.set_xlabel('ig')
    ax.set_ylabel('ip')
    plt.savefig(f'thetatb_scatter_box{ibox}.png')
    plt.close()

for ibox in range(0, 4):
    active= log.active[time_min:ntime,ibox]==1
    # ig_slice = slice(log.ig[time_min:ntime].min(), log.ig[time_min:ntime].max())
    # ip_slice = slice(log.ip[time_min:ntime,ibox].min(), log.ip[time_min:ntime,ibox].max())
    figure, ax = plt.subplots(1)
    c = np.arange(log.ig.size)
    (svtb.isel(ib=ibox, drop=True)).sel(ig=ig_slice, ip=ip_slice, drop=True).plot(ax=ax)
    ax.scatter(log.ig[time_min:ntime][active]+ log.fig[time_min:ntime][active], log.ip[time_min:ntime,ibox][active] + log.fip[time_min:ntime,ibox][active] ,c=c[time_min:ntime][active],cmap='jet')
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


