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

def pf2head(pf: np.ndarray) -> np.ndarray:
    return -10**(pf/10)

def head2pf(phead: np.ndarray) -> np.ndarray:
    return np.log10(-m2cm*phead)

# input
init_pF = np.array([2.2])
init_gwl = np.array([-6.0])
rch = np.array([0.001,0.0,0.003,0.0,0.0])

# box
box_area = np.array([10.0 * 10.0])
box_top = np.array([0.0, -5.0])
box_bot = np.array([-5.0, -8.0])
qrch = rch * box_area
box_qbot = np.zeros_like(box_top)
dtgw = 1.0
xi_theta = 1.0 # schaling factor

# current phead
phead = pf2head(init_pF)
# current table index
ip, fip = phead_to_index(phead)
ig, fig = gwl_to_index(init_gwl)


# first attempt

# get initial storage
sigmabtb = tabel.svtb[:,:,0] - dtgw * tabel.qmrtb[:,:] # first box

# first estimate based only on fig, not fip
sigmaln = sigmabtb[ip, ig] * fig*(sigmabtb[ip, ig + 1] - sigmabtb[ip, ig])

sigma_old = sigmaln
svold = tabel.svtb[ip, ig, 0] # box 1

# add recharge on top box 1
sigma = svold(1) + qrch * dtgw 

if (sigma - sigma_old) > dc:
    ipinc = 1
    if sigma > sigma_old:
        ipinc = -1
        
ip = np.max(ip + ipinc, tabel.ip[-1])
iprz  = np.min(ip,ip - ipinc)

sigmaln = sigmabtb[: , ig] * fig*(sigmabtb[: , ig + 1] - sigmabtb[:, ig])

if np.logical_and(sigma <= sigmaln[iprz], sigma >= sigmaln[iprz + 1])
    if sigmaln [ip + 1] - sigmaln[ip] > dc:
        fiprz = sigma - sigmaln[ip] / (sigmaln [ip + 1] - sigmaln [ip])
        fiprz = np.maximum(fiprz, 0.0)
        fiprz = np.maximum(fiprz,1.0)
    else:
        iprz  = iprz + 1 
        fiprx = 0.0
else:
    ip = ip + ipinc
    iprz = np.minimum(ip, ip - ipinc)

#               IF (iprz(b,k) .GE. nuip) THEN
#                 found(b)   = .TRUE.
#                 iprz(b,k)  = nuip - 1
#                 fiprz(b,k) = 1.0
#                  WRITE(iunerr,*) idtgw,'Przav at dry-end limit'
#               ELSEIF (iprz(b,k) .LE. nlip) THEN
#
#                 Situation with obstructed flow ! 
#                 > Position is finalized below (see @przmax)
#                 found(b)   = .TRUE.
#                 iprz(b,k)  = nlip
#                 fiprz(b,k) = 0.0
#               ENDIF




ph = np.arange(0.0,-1000.0,-0.01)
ip, fip = phead_to_index(ph)

gwl = np.arange(0.0,-10.0,-0.01)
ig, fig = gwl_to_index(gwl)


   
   
