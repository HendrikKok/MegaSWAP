import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import copy 



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
       data = {'value': tabel['dpgwtb'].fillna(0.0).to_numpy()
        }, index = np.arange(nxlig - 1 , nxuig + 1, 1)
       )

ig = np.arange(nxlig - 1 , nxuig + 1, 1)
ip = np.arange(nlip , nuip + 1, 1)
ib = np.arange(0,18,1)
svtb = tabel['svtb'].assign_coords({
    'ip': ip,
    'ig': ig,
    'ib': ib,
})

qmrtb = tabel['qmrtb'].assign_coords({
    'ip': ip,
    'ig': ig,
})

def get_ig_box_bottom(box_bottom_in:np.ndarray) -> np.ndarray:
    # perched conditions? 
    # TODO: 0.15 from database dpczsl
    box_bottom = box_bottom_in.copy()
    box_bottom[1] = box_bottom[1] - 0.15
    ig_box_bottom = np.full_like(box_bottom,-999,dtype = np.int32)
    ig_index = np.arange(1,dpgwtb.index[-1]+1,1)
    lower = -dpgwtb['value'][ig_index-1].to_numpy()
    upper = -dpgwtb['value'][ig_index].to_numpy()
    
    for bottom,index in zip(box_bottom,range(box_bottom.size)):
        ig_box_bottom[index] = ig_index[(lower > bottom) & (upper <= bottom)]
    return ig_box_bottom.astype(dtype=np.int32)

# ig_slrzbox -> ig-index of bottoms
def get_max_ig(ig,ib,ip,ig_box_bottom) -> int:
    igmax = ig
    if ig_box_bottom[ib] < ig:
        for igtp in range(ig,ig_box_bottom[ib],-1):
            if qmrtb.sel(ig=igtp,ip=ip) < 0.0:
                if qmrtb.sel(ig=igtp,ip=ip) < qmrtb.sel(ig=igmax,ip=ip):
                    igmax = igtp
    return igmax


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

def sigma2ip(sigma, ip, fip, ig, fig, ibox:int | None = None) -> tuple[int, int,float]:
    # if ibox > 0:
    #     return ip, fip
    #  sigma values could contain equal values?
    sigma1d = (sigmabtb.sel(ig=ig) + fig * (sigmabtb.sel(ig = ig + 1) - sigmabtb.sel(ig=ig)))
    sorter = np.argsort(sigma1d)
    ip_sigma1d = sorter[np.searchsorted(sigma1d, sigma, sorter = sorter)].item()
    ip = sigmabtb.ip[ip_sigma1d].item()

    # plt.plot([ibox, ibox], [sigma1d.min(),sigma1d.max()],'-o', label = f'sigmabtb_max_box{ibox}_ip{ip}')
    if (ip >= sigmabtb.ip.max()):
        print('out of bounds..')
        ip = sigmabtb.ip.max().item()
        fip = 0
        return ip, fip
    fip = ((sigma - sigma1d[ip_sigma1d]) / (sigma1d[ip_sigma1d + 1] - sigma1d[ip_sigma1d])).item()
    return ip, fip

def get_q(ip, fip, ig, fig) -> np.ndarray:
    qmr1d = (tabel['qmrtb'][:, ig] + fig * (tabel['qmrtb'][:, ig + 1] - tabel['qmrtb'][:, ig])).to_numpy().ravel()
    return qmr1d[ip] + fip * (qmr1d[ip + 1] - qmr1d[ip])

# Sv contains:
# svgwlnfu (msw1sgwln.for) -> liniair svtb value over ip range 
# msw1bd.for -> lineair vgwlnfu over ig-index

# msw1bd.for
#   qmv (nxb-1) = -(Sv(nxb) - Svold(nxb)) / dtgw
#   do b=nxb-2,1,-1
#     qmv(b) = -(Sv(b+1) - Svold(b+1)) / dtgw + qmv(b+1)

# msw1unsa.for
# sigma(b) = Svold(b) - qmv(b-1) * dtgw

# def get_sv_boxes(ig, fig, ip, fip, ib = None):
#     if ib is None:
#         ib_range = slice(svtb.ib[0],svtb.ib[-1])
#     else:
#         ib_range = ib
#     qmv_lineair = svtb.sel(ip=ip,ib = ib_range) + fip*(svtb.sel(ip=ip+1, ib = ib_range) - svtb.sel(ip=ip, ib=ib_range))
#     sv = qmv_lineair.sel(ig=ig) + fig * (qmv_lineair.sel(ig=ig+1) - qmv_lineair.sel(ig=ig))
#     return sv.to_numpy().ravel()
    
def get_qmv_bd(svnew: np.ndarray, svold: np.ndarray) -> np.ndarray:
    qmv = np.zeros_like(svnew)
    nxb = qmv.size
    qmv[nxb + 1] = -(svnew[nxb]-svold[nxb]) / dtgw
    for ibox in range(nxb-2,0,-1):
        qmv[ibox] = -(svnew[ibox + 1]-svold[ibox + 1]) / dtgw + qmv[ibox + 1]
    return qmv

def init_qmv(ig, fig, ip, fip):
    qmv_ip1 = qmrtb.sel(ig=ig,ip=ip) + fig * (qmrtb.sel(ig=ig+1,ip=ip) - qmrtb.sel(ig=ig,ip=ip))
    qmv_ip2 = qmrtb.sel(ig=ig,ip=ip + 1) + fig * (qmrtb.sel(ig=ig+1,ip=ip+1) - qmrtb.sel(ig=ig,ip=ip+1))
    return qmv_ip1 + fip * (qmv_ip2 - qmv_ip1)

def get_sv(ig, fig, ip, fip, ib=None):
    if ib is None:
        ib_range = slice(svtb.ib[0],svtb.ib[-1])
    else:
        ib_range = ib
    sv_lin = svtb.sel(ig=ig,ib = ib_range) + fig * (svtb.sel(ig=ig+1,ib = ib_range) - svtb.sel(ig=ig,ib = ib_range))
    return (sv_lin.sel(ip=ip) + fip * (sv_lin.sel(ip=ip+1) - sv_lin.sel(ip=ip))).to_numpy().ravel()


def get_qmv(svnew: np.ndarray, svold: np.ndarray, ibox,qin,qmv) -> np.ndarray:
    if ibox == 0:
        # return (svnew - svold) / dtgw - qin
        return qin + (svnew - svold) / dtgw
    else:
        return qmv[ibox - 1] + (svnew - svold) / dtgw

# input
init_pF = np.array([4.2])
init_gwl = np.array([-13.0])

box_area = 10.0 * 10.0
box_top = np.array([0.0, -5.0])
box_bot = np.array([-5.0, -8.0])

box_qbot = np.zeros_like(box_top)
dtgw = 1.0
xi_theta = 1.0 # schaling factor

# first attempt
nbox = 18
phead = np.zeros((nbox))
# first timestep at initial pF
phead[:] = pf2head(init_pF) 

box_bottom = np.array(
    [-1.000,
    -1.150,
    -5.000,
    -7.000,
    -10.00,
    -13.00,
    -16.00,
    -20.00,
    -25.00,
    -30.00,
    -35.00,
    -40.00,
    -45.00,
    -50.00,
    -60.00,
    -70.00,
    -80.00,
    -100.0,
    ]
)
ig_box_bottom = get_ig_box_bottom(box_bottom)

qrch_ar = np.array([0.003,0.002,0.003,0.004,0.005])
svold = np.zeros(nbox)
sv = np.zeros(nbox)

figure, ax = plt.subplots(5,figsize=(5,10)) # 5

for qrch, iplot in zip(qrch_ar,range(qrch_ar.size)):
    svold = np.zeros(nbox)
    sv = np.zeros(nbox)

    # first timestep at initial volume
    ig, fig = gwl_to_index(init_gwl)
    ip, fip = phead_to_index(pf2head(init_pF))

    # init sv
    svold = get_sv(ig,fig,ip,fip)

    # init qmv
    qmv = np.zeros(nbox)
    qmv[:] = init_qmv(ig,fig,ip,fip)
    
    itime = 0
    qin_list = []
    ip_list = []
    fip_list = []
    non_submerged_boxes = np.arange(nbox)[box_bottom > init_gwl]
    #plt.plot(np.full_like(non_submerged_boxes,init_pF), label = f'svtb_init_ip{ip}', color = 'grey')
    for ibox in non_submerged_boxes:
        print(f'box {ibox} and time {itime}')

        sigmabtb = svtb.sel(ib = ibox) - dtgw * qmrtb     # waarom veel kleiner dan de svold o.b.v. svtb
        ig, fig = gwl_to_index(init_gwl)
        ig = get_max_ig(ig, ibox, ip, ig_box_bottom) 
        # add recharge to get new volume
        if ibox == 0:
            qin = -qrch
        else:
            qin = -qmv[ibox - 1]
        qin_list.append(qin)
        sigma = svold[ibox] - qin * dtgw 
        # plt.plot(ibox, qin,'o', label = f'qin_{ibox}_ip{ip}')
        # plt.plot(ibox, sigma,'o', label = f'sigma_{ibox}_ip{ip}')
        # plt.plot(ibox, sigma,'o', label = f'sigma_{ibox}_ip{ip}')
        
        # ip vastzetten op rtzone?
        ip, fip = sigma2ip(sigma,ip,fip,ig,fig,ibox)
        phead[ibox] = ptb['value'][ip] + fip * (ptb['value'][ip + 1] - ptb['value'][ip])
        ip_list.append(ip)
        fip_list.append(fip)
        # ip, fip = phead_to_index(phead[ibox])
        sv[ibox] = get_sv(ig,fig,ip,fip,ibox)
        qmv[ibox] = get_qmv(sv[ibox], svold[ibox],ibox,qin,qmv)
    # p = ax[iplot].plot(qin_list, non_submerged_boxes, '-o', label = 'qin')
    ax[iplot].plot(qmv[non_submerged_boxes], non_submerged_boxes,'-o',label = 'quit') 
    #ax[iplot].plot(sv[non_submerged_boxes], non_submerged_boxes,'-o',label = 'quit') 
    # p = ax[0].plot(ip_list, non_submerged_boxes,'-o',label = 'ip')
    #ax[1].plot(fip_list, non_submerged_boxes,'-o',label = f'fip_rch{qrch}', color = p[0].get_color())
    #ax[iplot].plot(head2pf(phead[non_submerged_boxes]), non_submerged_boxes,'-o',label = 'phead') 
    ax[iplot].legend()
    ax[iplot].set_title(f'rch = {qrch:.4f}')
    # ax[iplot].set_xlim(0.,0.17)
    ax[iplot].invert_yaxis()
plt.legend()
plt.tight_layout()
plt.savefig("q.png")
plt.close()



pass