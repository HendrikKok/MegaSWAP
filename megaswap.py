import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

tabel = xr.open_dataset('database\\unsa_001.nc')
tabel['svtb'] = tabel['svtb'].where(tabel['svtb'] < 100.0).fillna(0.0)
tabel['qmrtb'] = tabel['qmrtb'].where(tabel['qmrtb'] < 100).fillna(0.0)

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
       data = {'index_gwtb': tabel.igdc.to_numpy().astype(dtype=np.int32)
        }, index = np.arange(tabel.igdcmn, tabel.igdcmx + 1, 1)
       )
dpgwtb = pd.DataFrame(
       data = {'value': tabel['dpgwtb'].to_numpy()
        }, index = np.arange(nxlig - 1 , nxuig + 1, 1)
       )

ptb_index = np.arange(nlip, nuip + 1)
ptb_values = np.zeros_like(ptb_index, dtype=np.float64)
ptb_values[ptb_index <= 0] = -ddpptb * ptb_index[ptb_index <= 0]
ptb_values[ptb_index > 0] = -cm2m*(10**(ptb_index[ptb_index > 0] * ddpptb))
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
    return igk.item(),figk.item()


def phead_to_index(ph: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    if not isinstance(ph, np.ndarray):
        ph = np.array([ph])
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
    return ip.item(), fip.item()


def pf2head(pf: np.ndarray) -> np.ndarray:
    return -10**(pf/10)

def head2pf(phead: np.ndarray) -> np.ndarray:
    return np.log10(-m2cm*phead)

def sigma2phead(sigma: np.ndarray, fig: np.ndarray, ig:np.ndarray) -> np.ndarray:
    # sigma values could contain equal values?
    sigma1d = (sigmabtb[:, ig] + fig * (sigmabtb[:, ig + 1] - sigmabtb[:, ig])).to_numpy().ravel()
    sorter = np.argsort(sigma1d)
    ip = sorter[np.searchsorted(sigma1d, sigma, sorter = sorter)]
    if (ip > sigma1d.size):
        raise ValueError('out of bounds sigmabtb')
    fip = (sigma - sigma1d[ip]) / (sigma1d[ip + 1] - sigma1d[ip])
    if not np.isfinite(fip):
        fip = 1.0
    phead_cm = ptb['value'][ip] + fip * (ptb['value'][ip + 1] - ptb['value'][ip])
    return phead_cm * cm2m

def get_q(ip, fip, ig, fig) -> np.ndarray:
    qlin = (tabel['qmrtb'][:, ig] + fig * (tabel['qmrtb'][:, ig + 1] - tabel['qmrtb'][:, ig])).to_numpy().ravel()
    return qlin[ip] + fip * (qlin[ip + 1] - qlin[ip])

# input
init_pF = np.array([2.2])
init_gwl = np.array([-6.0])
rch = np.array([0.0001,0.0002,0.0003,0.0001,0.00005])

# box
box_area = np.array([10.0 * 10.0])
box_top = np.array([0.0, -5.0])
box_bot = np.array([-5.0, -8.0])
qrch = rch * box_area
box_qbot = np.zeros_like(box_top)
dtgw = 1.0
xi_theta = 1.0 # schaling factor



# first attempt
nbox = 4
ntime = qrch.size
phead = np.zeros((nbox, ntime))
# first timestep at initial pF
phead[:,0] = pf2head(init_pF) 

svold = np.zeros((nbox, ntime))
# first timestep at initial volume
ig, fig = gwl_to_index(init_gwl)
ip, fip = phead_to_index(pf2head(init_pF) )
svold[:,0] = tabel.svtb[ip, ig, 0:nbox].to_numpy().ravel() 

q = np.zeros((nbox, ntime))

for itime in range(ntime):
    for ibox in range(nbox):
        print(f'box {ibox} and time {itime}')
        ig, fig = gwl_to_index(init_gwl) 
        ip, fip = phead_to_index(phead[ibox,itime])
        
        # get initial storage volume
        sigmabtb = tabel.svtb[:,:,ibox] - dtgw * tabel.qmrtb[:,:] # total volumebalans per box

        # add recharge to get new volume
        if ibox == 0:
            qin = -qrch[itime]
        else:
            qin = -q[ibox - 1, itime] # switch sign
        sigma = svold[ibox, itime] + qin * dtgw 

        # get new phead and indexes
        phead[ibox, itime] = sigma2phead(sigma, fig, ig)
        ip, fip = phead_to_index(phead[ibox, itime])
        
        # update q
        q[ibox, itime] = get_q(ip, fip, ig, fig)
    if itime + 1 < ntime:
        phead[:, itime + 1] = phead[:, itime]


pass