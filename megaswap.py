import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

tabel = xr.open_dataset('database\\unsa_001.nc')

mv = 0.0
bot = 13.0

cm2m = 1/100
m2cm = 100
ddpptb = tabel.ddpptb
ddgwtb = tabel.ddgwtb # -> nc
nuip = tabel.nuip
nuig = tabel.nuig
nlig   = tabel.nlig
nlip = tabel.nlip
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

ptb_index = np.arange(nlip, nuip + 1)
ptb_values = np.zeros_like(ptb_index, dtype=np.float64)
ptb_values[ptb_index <= 0] = -ddpptb * ptb_index[ptb_index <= 0]
ptb_values[ptb_index > 0] = -cm2m*(10**(ptb_index[ptb_index > 0]) * ddpptb)
ptb = pd.DataFrame(
    data = {'value': ptb_values},
    index = ptb_index
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


def phead_to_index(pF: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
  # positive ph = ponding -> so linear
  pFtb = -pF/ddpptb
  ip = pFtb.astype(dtype= np.int32) - 1  # int function for <0 rounds towards zero
  fip  = pFtb - ip
  # ph from zero to -1 cm 
  mask = pF < 0.0
  ip[mask] = 0
  fip[mask] = 0
  # ph from -1 cm onwards
  mask = pF < -1 * cm2m
  pFtb[mask] = np.log10(-m2cm*pF[mask])/ddpptb
  ip[mask] = np.minimum(pFtb[mask].astype(dtype=np.int32), nuip - 1)
  fip[mask] = pFtb[mask] - ip[mask]
  # min and maximize fractions
  fip = np.maximum(fip, 0.0)
  fip = np.minimum(fip, 1.0)
  return ip, fip


def pf2head(pf: np.ndarray) -> np.ndarray:
    return -10**(pf/10)

def head2pf(phead: np.ndarray) -> np.ndarray:
    return np.log10(-m2cm*phead)

def sigma2phead(sigma: np.ndarray, fig: np.ndarray, ig:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    sigma1d = (sigmabtb[:, ig] + fig * (sigmabtb[:, ig + 1] - sigmabtb[:, ig])).to_numpy().ravel()
    ip = np.searchsorted(sigma1d, sigma, sorter = np.argsort(sigma1d)) - 1
    if (ip > sigma1d.size):
        raise ValueError('out of bounds sigmabtb')
    fip = (sigma - sigma1d[ip]) / (sigma1d[ip + 1] - sigma1d[ip])
    phead_cm = ptb['value'][ip] + (fip * (ptb['value'][ip+1] - ptb['value'][ip]))
    return phead_cm * cm2m

# input
init_pF = np.array([2.2])
init_gwl = np.array([-6.0])
rch = np.array([0.0001,0.0,0.0003,0.0001,0.0])

# box
box_area = np.array([10.0 * 10.0])
box_top = np.array([0.0, -5.0])
box_bot = np.array([-5.0, -8.0])
qrch = rch * box_area
box_qbot = np.zeros_like(box_top)
dtgw = 1.0
xi_theta = 1.0 # schaling factor



# first attempt
ig, fig = gwl_to_index(init_gwl)
ip, fip = phead_to_index(pf2head(init_pF))
svold = tabel.svtb[ip, ig, 0].to_numpy().ravel()# box 1 -> for now initial volume; not sure about this!

## ---- do UNSA ---- ## box 1

# get initial storage volume
sigmabtb = tabel.svtb[:,:,0] - dtgw * tabel.qmrtb[:,:] # first box; sigma12tbfu
# sigmaln = sigmabtb[ip, ig] * fig*(sigmabtb[ip, ig + 1] - sigmabtb[ip, ig])
sigma_old = svold

# add recharge to get new volume
sigma = svold + qrch[0] * dtgw 

# get new phead and indexes
phead= sigma2phead(sigma, fig, ig)
ip, fip = phead_to_index(phead)



