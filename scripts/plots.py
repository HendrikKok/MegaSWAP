import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

tabel = xr.open_dataset('unsa_001.nc')

mv = 0.0
bot = 13.0

cm2m = 1/100
m2cm = 100
ddpptb = tabel.ddpptb
ddgwtb = tabel.ddgwtb # -> nc
nuip = 27 # tabel.ip.size
nuig = 54 # tabel.ig.size
nlig   = -1 # hard-wired value of -1
nxlig = tabel.nxlig
nxuig = tabel.nuig
dc = 0.1e-6

igdc = pd.DataFrame(
       data = {'index_gwtb': tabel.igdc.to_numpy()
        }, index = np.arange(tabel.igdcmn, tabel.igdcmx + 1, 1)
       )
dpgwtb = pd.DataFrame(
       data = {'value': tabel['dpgwtb'].to_numpy()
        }, index = np.arange(nxlig - 1 , nxuig + 2, 1)
       )


def gwl_to_index(gwl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  below_bot = gwl < (mv - bot)
  gwl[below_bot] = mv - bot
  dpgw = mv - gwl
  igk = igdc['index_gwtb'][((dpgw + dc) / ddgwtb).astype(dtype=np.int32)].to_numpy()  # ddgwtb = stepsize in igdc array
  # ponding
  ponding = dpgw < 0.0
  igk[ponding] = igk[ponding]  - 1
  # maximize on array size?
  igk = np.maximum(igk, nlig - 1)
  figk = (dpgw - dpgwtb['value'][igk].to_numpy()) / (dpgwtb['value'][igk + 1].to_numpy() - dpgwtb['value'][igk].to_numpy())
  return igk,figk


def phead_to_index(ph: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
  # positive ph = ponding -> so linear
  pFtb = -ph/ddpptb
  ip = pFtb.astype(dtype= np.int32) - 1  # int function for <0 rounds towards zero
  fip  = pFtb - ip
  # ph from zero to -1 cm 
  mask = ph < 0.0
  ip[mask] = 0
  fip[mask] = 0
  # ph from -1 cm onwards
  mask = ph < -1 * cm2m
  pFtb[mask] = np.log10(-m2cm*ph[mask])/ddpptb
  ip[mask] = np.minimum(pFtb[mask].astype(dtype=np.int32), nuip - 1)
  fip[mask] = pFtb[mask] - ip[mask]
  # min and maximize fractions
  fip = np.maximum(fip, 0.0)
  fip = np.minimum(fip, 1.0)
  return ip, fip


ph = np.arange(0.0,-1000.0,-0.01)
ip, fip = phead_to_index(ph)

gwl = np.arange(0.0,-10.0,-0.01)
ig, fig = gwl_to_index(gwl)

# controle plots
plt.plot(np.log10(-m2cm*ph), label = 'pF')
plt.plot(ip, label = 'ip')
plt.plot(fip, label = 'fip')
plt.legend()
plt.tight_layout()
plt.savefig("index_p.png")
plt.close()

plt.plot(np.arange(ig.shape[0]),ig, label = 'ig')
plt.plot(np.arange(fig.shape[0]),fig, label = 'fig')
plt.legend()
plt.tight_layout()
plt.savefig("index_h.png")
plt.close()

head = mv - (dpgwtb['value'][ig].to_numpy() + (fig*(dpgwtb['value'][ig + 1].to_numpy() - dpgwtb['value'][ig].to_numpy())))
plt.plot(head, label = 'table',linewidth=2.0)
plt.plot(gwl, label = 'input',linewidth=1.0)
plt.legend()
plt.tight_layout()
plt.savefig("head_head.png")
plt.close()


ip = tabel.ip.to_numpy()
ig = tabel.ig.to_numpy()
ib = tabel.ib.to_numpy()
flux = xr.DataArray(
    tabel['qmrtb'].to_numpy().reshape((ip.size,ig.size)),
    coords = {
        'ip': ip,
        'ig': ig,
    },
    dims = ['ip','ig']
)

figure, ax = plt.subplots(1,subplot_kw={"projection": "3d"})
flux.plot.surface(ax = ax)
#ax.axes.set_aspect('equal')
# plt.tight_layout()
plt.savefig("flux.png")
plt.close()

storage = xr.DataArray(
    data = tabel['svtb'].to_numpy().reshape((ip.size,ig.size, ib.size)),
    coords = {
        'ip': ip,
        'ig': ig,
        'ib': ib,
    },
    dims = ['ip','ig','ib']
)
for b in ib:
    figure, ax = plt.subplots(1,subplot_kw={"projection": "3d"})
    storage.sel(ib=b).plot.surface(ax = ax, x = 'ig', y = 'ip', yincrease = False)
    #ax.axes.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"storage_box{b}.png")
    plt.close()