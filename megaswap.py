import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import copy 



tabel = xr.open_dataset('database\\unsa_002.nc')
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
sc1min = 1.0e-03
sc1_default = 0.15
rootzone_dikte = 0.5


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
    return ip, fip


def pf2head(pf: np.ndarray) -> np.ndarray:
    return -10**(pf/10)

def head2pf(phead: np.ndarray) -> np.ndarray:
    return np.log10(-m2cm*phead)

def sigma2ip(sigma, ig, fig, ibox:int) -> tuple[int,float]:
    #  sigma values could contain equal values?
    sigmabtb = svtb.sel(ib = ibox) - dtgw * qmrtb
    sigma1d = (sigmabtb.sel(ig=ig) + fig * (sigmabtb.sel(ig = ig + 1) - sigmabtb.sel(ig=ig)))
    sorter = np.argsort(sigma1d)
    ip_sigma1d = sorter[np.searchsorted(sigma1d, sigma, sorter = sorter)].item()
    ip = sigmabtb.ip[ip_sigma1d].item()

    # plt.plot([ibox, ibox], [sigma1d.min(),sigma1d.max()],'-o', label = f'sigmabtb_max_box{ibox}_ip{ip}')
    if (ip >= sigmabtb.ip.max()):
        print('out of bounds..')
        ip = sigmabtb.ip.max().item() - 1
        fip = 1.0
        return ip, fip
    fip = ((sigma - sigma1d[ip_sigma1d]) / (sigma1d[ip_sigma1d + 1] - sigma1d[ip_sigma1d])).item()
    return ip, fip

def qmr2ip(qmr, ig, sigmabtb) -> tuple[int,float]:
    qmr1d = qmrtb.sel(ig=ig)
    sorter = np.argsort(qmr1d)
    ip_qmr1d = sorter[np.searchsorted(qmr1d, qmr, sorter = sorter)].item()
    ip = qmr1d.ip[ip_qmr1d].item()
    if (ip >= sigmabtb.ip.max()):
        print('out of bounds..')
        ip = sigmabtb.ip.max().item()
        fip = 0
        return ip, fip
    fip = (qmr - qmr1d.sel(ip=ip -1).item()) / (qmr1d.sel(ip=ip).item() - qmr1d.sel(ip=ip-1).item())
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


def update_sc1(lvgw, lvgw_old, s, sold, prz, sgwln):
    ig_mf6, _ = gwl_to_index(lvgw)
    treshold = 0.00025
    sgwln[ig] = msw1sgwlnkig(ig, ig_old, prz, lvgw_old)
    if abs(lvgw - lvgw_old) > treshold:
        # sc1 waterbalance 
        if lvgw > mv or lvgw_old > mv:
            sc1_wb = (s - sold) / (lvgw - lvgw_old)
        else:
            sgwln[ig] = msw1sgwlnkig(ig, ig_old, prz, lvgw_old)
            sgwln[ig + 1] = msw1sgwlnkig(ig + 1, ig_old, prz, lvgw_old)
            sc1_wb = (sgwln[ig] - sgwln[ig + 1])/ (dpgwtb.loc[ig + 1] - dpgwtb.loc[ig])
        sc1_wb = np.maximum(sc1_wb,sc1min)
        sc1_wb = np.minimum(sc1_wb,1.0)
        # sc1 waterlevel
        #   sgwln = msw1sgwlnkig(ig_mf6, ig_old, prz, lvgw_old)
        #   if ig_mf6 <= nuig and abs(sgwln[ig_mf6+1]) < dc:
        #       sgwln = msw1sgwlnkig(ig_mf6 + 1, ig_old, prz, lvgw_old)
        #   sc1_lv = sgwln[ig_mf6] - sgwln[ig_mf6+1]/(dpgwtb.sel(ig = ig_mf6 + 1) - dpgwtb.sel(ig=ig_mf6))
        #   sc1_lv = max(sc1_lv,sc1min)
        #   sc1_lv = min(sc1_lv,1.0)
        #   # combined
        #   sc1 = 0.5 * (sc1_wb + sc1_lv)
        sc1 = sc1_wb
        sc1 = np.maximum(sc1,sc1min)
    elif lvgw > mv:
        sc1 = 1.0
    else:
        # no change, use trajectory value
        sgwln[ig_mf6] = msw1sgwlnkig(ig_mf6, ig_old, prz, lvgw_old)
        sgwln[ig_mf6 + 1] = msw1sgwlnkig(ig_mf6 + 1, ig_old, prz, lvgw_old)
        sc1 = (sgwln[ig_mf6] - sgwln[ig_mf6 + 1])/(dpgwtb.loc[ig_mf6+1] - dpgwtb.loc[ig_mf6])
    sc1 = np.maximum(sc1,sc1min)
    sc1 = np.minimum(sc1,1.0)
    return sc1, sgwln

def msw1sgwlnkig(ig, ig_old, prz, lvgw_old): 
    dpgwold = mv - lvgw_old
    sgwln= 0.0 
    for ibox in non_submerged_boxes:
        sgwln += svgwlnfu(ibox, ig, ig_old, prz[ibox], dpgwold)
    return sgwln

def svgwlnfu(ibox, ig, igold, prz, dpgwold):
    sigmabtb = svtb.sel(ib = ibox) - dtgw * qmrtb
    iprz, fiprz = phead_to_index(prz)
    # current situation ('from') 
    peq    = -dpgwold + 0.5 * rootzone_dikte  #  equilibrium pressure head for the currentn groundwater level
    # new situation ('to')
    peqtb  = -dpgwtb['value'][ig] + 0.5 * rootzone_dikte  #  presure head from position in the dpgwtb-table.
    # only for deeper table positions ig than the current one igkold
    # only if the current situation does not have plocking
    # only if the 'to' situation is percolation
    # only if the current groundwater level is above the bottom of the box
    if ig > igold and prz > peqtb:
        if dpgwold < 0.05 or prz < peq:
            # set to equilibrium value; from cap-rise to percolation 
            pnew = peqtb
            ip, fip = phead_to_index(pnew)
        elif prz > peq:
            # semi-implicit scheme; from percolation to percolation
            ip = iprz  
            fip = fiprz
            qmvtp = qmrtb.sel(ig=ig,ip=ip).item() + fip * (qmrtb.sel(ig = ig, ip=ip+1).item() - qmrtb.sel(ig=ig, ip= ip).item())
            # average ; allow for an increase of the percolation
            if qmvtp < 0.0 and qmvtp < qmv[ibox]:
                qmvtp = qmvtp*0.5 + qmv[ibox]*0.5
            # from qmvtp find phead and indexes
            ip, fip = qmr2ip(qmvtp, ig, sigmabtb)
            pnewtp = ptb['value'][ip] + fip * (ptb['value'][ip + 1] - ptb['value'][ip])
            pnew  = prz - (dpgwtb['value'][ig] - dpgwold)
            if pnew > pnewtp:
                # update index based on maximum dragdown 
                ip, fip = phead_to_index(pnew)
            else:
                ip = iprz
                fip = fiprz
        else:
            ip = iprz
            fip = fiprz
    else:
        ip = iprz
        fip = fiprz
    return svtb.sel(ib = ibox, ig=ig, ip=ip).item() + fip * (svtb.sel(ib = ibox, ig=ig, ip=ip + 1).item()  - svtb.sel(ib = ibox, ig=ig, ip=ip).item())


def f_smoothing(iter):
    iterur1 = 3
    iterur2 = 5
    if iter <= iterur1:
        return 1.0
    elif iter > iterur2:
        return 0.0
    else:
        omegalv = 1.0 - 1.0*(iter - iterur1)/(iterur2 - iterur1)
        return max(omegalv,0.0)



# input
init_pF = np.array([1.2])
init_gwl = np.array([-6.0])

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

qrch_ar = np.array([0.003,0.002,0.003,0.004,0.005, 0.0, 0.005,0.006,0.001,0.001,0.002])
svold = np.zeros(nbox)
sv = np.zeros(nbox)
qmv = np.zeros(nbox)

non_submerged_boxes = np.arange(nbox)[box_bottom > init_gwl]
q_out = np.full((qrch_ar.size +1,non_submerged_boxes.size +1),np.nan) # ntime, nbox


def update_unsa(ig, phead, qrch)-> tuple[np.ndarray,np.ndarray]:
    for ibox in non_submerged_boxes:
        # ig_local = get_max_ig(ig, ibox, ip[ibox], ig_box_bottom) 
        # add recharge to get new volume
        if ibox == 0:
            qin = -qrch
        else:
            qin = -qmv[ibox - 1]
        sigma = svold[ibox] - qin * dtgw 
        ip, fip = sigma2ip(sigma, ig, fig, ibox)
        phead[ibox] = ptb['value'][ip] + fip * (ptb['value'][ip + 1] - ptb['value'][ip])
        # ip, fip = phead_to_index(phead[ibox])
        sv[ibox] = get_sv(ig,fig,ip,fip,ibox)
        qmv[ibox] = get_qmv(sv[ibox], svold[ibox], ibox, qin, qmv)
    return phead, sv, qmv

def summed_sv(sv):
    s = 0.0
    for ibox in non_submerged_boxes:
        s +=sv[ibox]
    return s

gwl = init_gwl
gwl_old = gwl
phead = np.full(nbox, pf2head(init_pF))
s_old = 0

ig, fig = gwl_to_index(gwl)
ip,fip = phead_to_index(phead)
for ibox in range(nbox):
    svold[ibox] = get_sv(ig,fig,ip[ibox],fip[ibox], ibox)      

sgwln = np.full(svtb.ig.size, -999.0)
for qrch, itime in zip(qrch_ar,range(qrch_ar.size)):
    ip_old, fip_old = phead_to_index(phead)
    sgwln[:] = 999.0
    sc1_list = []
    vsim_list = []
    ip_list = []
    for iter in range(5):
        ig, fig = gwl_to_index(gwl)
        phead, sv, qmv = update_unsa(ig, phead, qrch)
        s = summed_sv(sv)
        s_old = summed_sv(svold)
        vsim = qrch - (s - s_old) / dtgw
        vsim_list.append(vsim) 
        if iter == 0:
            # exchange to modflow
            # vsim + sc1_default
            pass
        elif iter > 0:
            sc1, sgwln = update_sc1(gwl, gwl_old, s, s_old, phead, sgwln)
            sc1_list.append(sc1)

        ig_old = ig
        gwl_old = gwl
        phead_old = phead
    svold = sv
        
    




pass