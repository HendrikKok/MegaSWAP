import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

tabel = xr.open_dataset('database\\unsa_300.nc')
tabel['svtb'] = tabel['svtb'].fillna(0.0)
tabel['qmrtb'] = tabel['qmrtb'].fillna(0.0)
# tabel['svtb'] = tabel['svtb'].where(tabel['svtb'] > 100.0).fillna(0.0)
# tabel['qmrtb'] = tabel['qmrtb'].where(tabel['qmrtb'] > 100).fillna(0.0)

mv = 0.0
bot = 99.0

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
rootzone_dikte = 1.0


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




# figure, ax = plt.subplots(1,subplot_kw={"projection": "3d"})
# qmrtb.plot.surface(ax = ax)
# #ax.axes.set_aspect('equal')
# # plt.tight_layout()
# plt.savefig("flux.png")
# plt.close()
# 
# for b in ib:
#     figure, ax = plt.subplots(1,subplot_kw={"projection": "3d"})
#     svtb.sel(ib=b).plot.surface(ax = ax, x = 'ig', y = 'ip', yincrease = False)
#     #ax.axes.set_aspect('equal')
#     plt.tight_layout()
#     plt.savefig(f"plots/storage_box{b}.png")
#     plt.close()

def get_ig_box_bottom(box_bottom_in:np.ndarray) -> np.ndarray:
    # perched conditions? 
    # TODO: 0.15 from database dpczsl
    box_bottom = box_bottom_in.copy()
    # box_bottom[1] = box_bottom[1] - tabel.dpczsl
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
    igk = igdc['index_gwtb'][np.floor((dpgw + dc) / ddgwtb)].to_numpy()  # ddgwtb = stepsize in igdc array
    # ponding
    ponding = dpgw < 0.0
    igk[ponding] = igk[ponding] - 1
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

def sigma2ip(sigma, ig, fig, ibox:int, sigma_list) -> tuple[int,float]:
    sigmabtb = svtb.sel(ib = ibox) - dtgw * qmrtb  
    sigma1d = (sigmabtb.sel(ig=ig) + fig * (sigmabtb.sel(ig = ig + 1) - sigmabtb.sel(ig=ig)))
    sorter = np.argsort(sigma1d)
    sorted_index = np.searchsorted(sigma1d, sigma, sorter = sorter)
    if sorted_index >= sorter.size:
        ip_index = sigmabtb.ip.max()
    else:
        ip_index = sorter[sorted_index].item()
    if ip_index >= sigmabtb.ip.max():
        print('out of max bounds..')
        ip = sigmabtb.ip.max().item() - 1
        fip = 1.0
    elif ip_index < sigmabtb.ip.min():
        ip = sigmabtb.ip.min()
        fip = 0
    else:
        ip = sigmabtb.ip[ip_index].item()
        fip = ((sigma - sigma1d[ip_index]) / (sigma1d[ip_index + 1] - sigma1d[ip_index])).item()
    if sigma_list is not None:
        sigma_list[itime,ibox,:] = sigma1d.to_numpy()
    return ip, fip


def sv2ip(sv, ig, fig, ibox:int, sigma_list) -> tuple[int,float]:
    sigmabtb = svtb.sel(ib = ibox) - dtgw * qmrtb  
    sigma1d = (sigmabtb.sel(ig=ig) + fig * (sigmabtb.sel(ig = ig + 1) - sigmabtb.sel(ig=ig)))
    sorter = np.argsort(sigma1d)
    sorted_index = np.searchsorted(sigma1d, sv, sorter = sorter)
    if sorted_index >= sorter.size:
        ip_index = sigmabtb.ip.max()
    else:
        ip_index = sorter[sorted_index].item()
    if ip_index >= sigmabtb.ip.max():
        print('out of max bounds..')
        ip = sigmabtb.ip.max().item() - 1
        fip = 1.0
    elif ip_index < sigmabtb.ip.min():
        ip = sigmabtb.ip.min()
        fip = 0
    else:
        ip = sigmabtb.ip[ip_index].item()
        fip = ((sv - sigma1d[ip_index]) / (sigma1d[ip_index + 1] - sigma1d[ip_index])).item()
    if sigma_list is not None:
        sigma_list[itime,ibox,:] = sigma1d.to_numpy()
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

def get_sv(ig, fig, ip, fip, sv_lin_list,ib=None,):
    if ib is None:
        ib_range = slice(svtb.ib[0],svtb.ib[-1])
    else:
        ib_range = ib
    sv_lin = svtb.sel(ig=ig,ib = ib_range) + fig * (svtb.sel(ig=ig+1,ib = ib_range) - svtb.sel(ig=ig,ib = ib_range))
    if sv_lin_list is not None:
        sv_lin_list[ibox,:] = sv_lin #debug
    return (sv_lin.sel(ip=ip) + fip * (sv_lin.sel(ip=ip+1) - sv_lin.sel(ip=ip))).to_numpy().ravel().item()

def get_qmv(svnew: np.ndarray, svold: np.ndarray, ibox,qin,qmv_in) -> np.ndarray:
    qmv = np.copy(qmv_in)
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
    return sgwln.item()

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


def update_unsa(ig, fig, phead, qrch, sv, svold, sv_lin_list, sigma_list,sigma_ar):
    qmv[:] = 0.0
    ip, _ = phead_to_index(phead)
    for ibox in non_submerged_boxes:
        ig_local = get_max_ig(ig, ibox, ip[ibox], ig_box_bottom) 
        if ig != ig_local:
            pass
        # add recharge to get new volume
        if ibox == 0:
            qin = -qrch
        else:
            qin = qmv[ibox - 1]
        sigma = svold[ibox] - qin * dtgw 
        if sigma_ar is not None:
            sigma_ar[itime,ibox] = sigma
        ip[ibox], fip[ibox] = sigma2ip(sigma, ig_local, fig, ibox, sigma_list)
        phead[ibox] = ptb['value'][ip[ibox]] + fip[ibox] * (ptb['value'][ip[ibox] + 1] - ptb['value'][ip[ibox]])
        ip_ar[itime,ibox] = ip[ibox] # debug
        fip_ar[itime,ibox] = fip[ibox] # debug 
        sv[ibox] = get_sv(ig_local,fig,ip[ibox],fip[ibox],sv_lin_list,ibox)
        qmv[ibox] = get_qmv(sv[ibox], svold[ibox], ibox, qin, qmv)
    return phead, sv, qmv, ip, fip

def finalize_unsa(ig,fig,ig_init,fig_init,ip,fip,qmodf,sv_lin_ar):
    # based on msw1bd
    # update sv based on new pressure head
    ibmax = nbox - 1 # 0-based
    #TODO: fix issue with new sv's
    for ibox in range(ibmax):
        ig_local = get_max_ig(ig, ibox, ip[ibox], ig_box_bottom) 
        sv[ibox] = get_sv(ig_local,fig,ip[ibox],fip[ibox],sv_lin_ar,ibox)
    # update qmv's
    qmv[:] = 0.0
    qmv[ibmax - 1] = -(sv[ibmax] - svold[ibmax]) / dtgw + qmodf
    if qmv[0] > 0.0:
        raise ValueError('inflow box 1 from bottom')
    for ibox in range(ibmax - 2,-1,-1):
        qmv[ibox] = -(sv[ibox + 1] - svold[ibox + 1]) / dtgw + qmv[ibox+1]
    qmv[ibmax] = (sv[ibmax] - svold[ibmax]) / dtgw + qmv[ibmax - 1]
    # update prz
    pass
    for ibox in non_submerged_boxes:
        # if ibox == 0:
        #     qin = -qrch
        # else:
        #     qin = qmv[ibox - 1]
        # sigma = svold[ibox] - qin * dtgw
        # ig_local = get_max_ig(ig_init, ibox, ip[ibox], ig_box_bottom) 
        ip[ibox], fip[ibox] = sv2ip(sv[ibox], ig_local, fig, ibox, None)
        phead[ibox] = ptb['value'][ip[ibox]] + fip[ibox] * (ptb['value'][ip[ibox] + 1] - ptb['value'][ip[ibox]])
    return sv, qmv, ip, fip, phead

def summed_sv(sv):
    s = 0.0
    for ibox in non_submerged_boxes:
        s +=sv[ibox]
    return s

def get_unsa_heads(sarg, sgwln, prz, lvgw_old, ig_start):
    ig = ig_start
    sgwln[ig] = msw1sgwlnkig(ig, ig_old, prz, lvgw_old)
    sgwln[ig + 1] = msw1sgwlnkig(ig, ig_old, prz, lvgw_old)
    dif = sgwln[ig+1] - sarg
    ig = None
    if dif < 0.0:
        for ig in range(ig_start, -1, -1):
            sgwln[ig] = msw1sgwlnkig(ig, ig_old, prz, lvgw_old)
            if sarg <= sgwln[ig] and sarg >= sgwln[ig+1]:
                break
    else:
        for ig in range(ig_start, 51, 1):
            sgwln[ig + 1] = msw1sgwlnkig(ig + 1, ig_old, prz, lvgw_old)
            if sarg <= sgwln[ig] and sarg >= sgwln[ig+1]:
                break
    fig = (sarg-sgwln[ig])/(sgwln[ig+1]-sgwln[ig])
    return mv - (dpgwtb.loc[ig] + fig*(dpgwtb.loc[ig+1]-dpgwtb.loc[ig])), ig, fig

def get_non_submerged_boxes(top_boxes, ig):
    mask = dpgwtb['value'].loc[ig] >= -top_boxes
    return np.arange(top_boxes.size)[mask]

def get_plock(phead):
    if phead[0] <= phead[1] or tabel.dpczsl < dc:
        return True
    return False



# input
init_pF = np.array([1.2])
init_gwl = np.array([-6.9])

box_area = 10.0 * 10.0
box_top = np.array([0.0, -5.0])
box_bot = np.array([-5.0, -8.0])

box_qbot = np.zeros_like(box_top)
dtgw = 1.0

# first attempt
nbox = 18
box_bottom = np.array(
    [-rootzone_dikte,
    -1.0 - tabel.dpczsl,
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
box_top = np.zeros_like(box_bottom)
box_top[1:] = box_bottom[1:] - np.diff(box_bottom)

ig_box_bottom = get_ig_box_bottom(box_bottom)

# qrch_ar = np.array([0.003,0.002,0.003,0.004,0.005, 0.0, 0.005,0.006,0.001,0.001,0.002])
qrch_ar = np.array([0.0016]*160)  #0.0016
svold = np.zeros(nbox)
sv = np.zeros(nbox)
qmv = np.zeros(nbox)

gwl_unsa = np.full_like(qrch_ar,init_gwl)

gwl = init_gwl 
gwl_old = np.copy(gwl)

s_old = 0
ig, fig = gwl_to_index(gwl)
ig_old = np.copy(ig)


non_submerged_boxes = get_non_submerged_boxes(box_top, ig)
q_out = np.full((qrch_ar.size +1,non_submerged_boxes.size +1),np.nan) # ntime, nbox
phead_out = np.full((qrch_ar.size +1,nbox),np.nan) # ntime, nbox


phead = np.full(nbox, 0.0)
z = box_top # - (box_top - box_bottom) / 2.
# mask = dpgwtb['value'].loc[ig] > -box_top
phead[:] = -(z-init_gwl)[0]
phead_init = np.copy(phead)


sv_lin_ar = np.zeros((qrch_ar.size,nbox,svtb.ip.size))
ip,fip = phead_to_index(phead)
for ibox in range(nbox):
    svold[ibox] = get_sv(ig,fig,ip[ibox],fip[ibox],sv_lin_ar[0,:,:],ibox)      

sgwln = np.full(svtb.ig.size, -999.0)
vsim_list = []
sc1_list = []
s_list = []
sold_list = []
sdif_list = []
tig_list = []
tfig_list = []
gwl_list = []
qmv_ar = np.zeros((qrch_ar.size,nbox))
sv_ar = np.zeros_like(qmv_ar)
fip_ar = np.zeros_like(qmv_ar)
ip_ar =np.zeros_like(qmv_ar)
ip_ar2 = np.zeros_like(qmv_ar)
rhs = np.zeros_like(qrch_ar)
lhs = np.zeros_like(qrch_ar)
qmodf = np.zeros_like(qrch_ar)
ibox_list = []
sigma_lin_ar = np.full((qrch_ar.size,nbox, svtb.ip.size), np.nan)  # time, box, nip
sigma_ar = np.zeros((qrch_ar.size,nbox))
phead_old = np.copy(phead)

for qrch, itime in zip(qrch_ar,range(qrch_ar.size)):
    gwl_list.append(gwl[0])
    sgwln[:] = 999.0
    for iter in range(1):
        ig, fig = gwl_to_index(gwl)
        if itime >= 0:
            pass
            # gwl = np.array([gwl_unsa[itime -1]])
            non_submerged_boxes = get_non_submerged_boxes(box_top, ig)
            ibox_list.append(non_submerged_boxes.size)
        tig_list.append(ig)
        tfig_list.append(fig)
        phead, sv, qmv, ip, fip = update_unsa(ig, fig, phead_old, qrch, sv, svold, sv_lin_ar[itime,:,:], sigma_lin_ar,sigma_ar)
        nnbox = non_submerged_boxes.size - 1
        qmv_ar[itime,0:nnbox] = qmv[0:nnbox]
        sv_ar[itime,0:nnbox] = sv[0:nnbox]
        
        s = summed_sv(sv)
        s_old = summed_sv(svold)
        vsim = qrch - (s - s_old) / dtgw
        vsim_list.append(vsim) 
        if iter == 0:
            # exchange to modflow
            # vsim + sc1_default
            pass
        elif iter > 0:
            pass
        if itime < 100 and itime > 0:
            pass
            gwl[0] = gwl[0] - 0.05 # update to new heads 0.05
        sc1, sgwln = update_sc1(gwl, gwl_old, s, s_old, phead, sgwln)

            
        #gwl_unsa[itime], ig2, fig2 = get_unsa_heads(s, sgwln, phead, gwl_old, ig)
        #gwl[0] = gwl_unsa[itime] # debug .
        gwl_unsa[itime] = gwl[0]
        

        qmodf[itime] = (sc1 * (gwl_old - gwl) - vsim) # doet nog niet veel op phead
        
        rhs[itime] = qmodf[itime] + vsim   # plot
        lhs[itime] = sc1 * (gwl_old - gwl) # plot
        ig2, fig2 = gwl_to_index(gwl)
        sv, qmv, ip, fip, phead = finalize_unsa(ig2,fig2,ig,fig,ip,fip, qmodf[itime],None)
        
        # old = new
        gwl_old = np.copy(gwl)
        phead_old = np.copy(phead)
        svold = np.copy(sv)
        ig_old = np.copy(ig2)
        ip_old, fip_old = np.copy(ip), np.copy(fip)
        
        # log
        sc1_list.append(sc1)
        s_list.append(s)
        sold_list.append(s_old)
        sdif_list.append(s_old-s)
        nn = non_submerged_boxes.size
        phead_out[itime,0:nn] = phead[0:nn]
    
    
# plotting

# plot 1
max_box = 4
# figure, ax = plt.subplots(1,2)

figure, ax = plt.subplot_mosaic("""
                                00113
                                22113
                                """)   # [[0,1,3],[2,1,3]]
ax['1'].plot(phead_init[0:max_box], z[0:max_box], color= 'black')
n = int(qrch_ar.size/10)
# n=1
colors = []
for itime in range(0,qrch_ar.size,n):
    p = np.repeat(phead_out[itime,0:max_box],2)
    y = np.stack([box_top[0:max_box],box_bottom[0:max_box]],axis=1).ravel()
    plot = ax['1'].plot(p, y, label = f"t={itime}")
    colors.append(plot[0].get_color())
pmin = phead_out[np.isfinite(phead_out)].min()
pmax = phead_out[np.isfinite(phead_out)].max()
ax['1'].hlines(0.0,pmin,pmax, color='grey')
for ibox in range(max_box):
    ax['1'].hlines(box_bottom[ibox],pmin,pmax, color='grey')
# ax['1'].legend()

ax['3'].hlines(0.0,0,1, color='grey')
for ibox in range(max_box):
    ax['3'].hlines(box_bottom[ibox],0,1, color='grey')
# ax['3'].hlines(init_gwl,0,1,color='blue')

icol = 0
for itime in range(0,qrch_ar.size,n):
# for head in gwl_list:
    head = gwl_list[itime]
    ax['3'].hlines(head,0,1, color= colors[icol], label = f"t={itime}")
    icol+=1
ax['3'].legend()


# ax[1].set_xlim(-8, pmax)
# ax[0].set_ylim(-8, pmax)
for ibox in range(max_box):
    ax['0'].plot(phead_out[:,ibox], label = f"h{ibox}")
ax['0'].legend()

ax['2'].plot(ibox_list, label = 'active boxes')
ax['2'].legend()
plt.tight_layout()
plt.savefig(f"pheads_dtgw_{dtgw}.png")
plt.close()


figure, ax = plt.subplots(1)
ax.plot(qrch_ar, label = 'pp')
ax.plot(vsim_list, label = 'qunsa')
ax.legend()
plt.tight_layout()
plt.savefig(f"recharge_dtgw_{dtgw}.png")
plt.close()

# figure, ax = plt.subplots(2)
# ax[0].plot(sold_list, label = 'svold')
# ax[0].plot(s_list, label = 's')
# 
# ax[1].plot(sdif_list, label = 'sv')
# ax[0].legend()
# plt.tight_layout()
# plt.savefig(f"dsv_dtgw_{dtgw}.png")
# plt.close()


figure, ax = plt.subplots(2)
nn = qrch_ar.size - 1
ax[0].plot(sc1_list, label = 'sc1')
ax[1].plot(gwl_unsa, label = 'gwl')
# ax[1].hlines(0.0,0,nn, color='grey')
# for ibox in range(max_box):
#     ax[1].hlines(box_bottom[ibox],0,nn, color='grey')
ax[1].legend()
ax[0].legend()
plt.tight_layout()
plt.savefig(f"mf6_dtgw_{dtgw}.png")
plt.close()



figure, ax = plt.subplots(2)
nn = qrch_ar.size - 1
for ibox in range(4):
    ax[0].plot(sv_ar[:,ibox], label = f'sv_{ibox}')
    ax[1].plot(qmv_ar[:,ibox], label = f'qmv_{ibox}')
    # ax[2].plot(fip_ar[:,ibox], label = f'fip_{ibox}')
    # ax[3].plot(ip_ar[:,ibox], label = f'ip_{ibox}')
ax[0].legend()
ax[1].legend()
# ax[2].legend()
# ax[3].legend()
plt.tight_layout()
plt.savefig(f"ds_{dtgw}.png")
plt.close()


figure, ax = plt.subplots(4)
ax[0].plot(tig_list, color='red')
ax[1].plot(tfig_list, color='red')

tig = np.array(tig_list)
fig = np.array(tfig_list)


ax[2].plot(mv - (dpgwtb['value'][tig].to_numpy() + fig*(dpgwtb['value'][tig+1].to_numpy()-dpgwtb['value'][tig].to_numpy())), label = 'dpgw')
ax[2].plot(gwl_unsa, label = 'gwl_unsa')
ax[2].legend()
ax[3].plot(gwl_list)

plt.tight_layout()
plt.savefig(f"igfig_{dtgw}.png")
plt.close()


ntime, nbox, _ = sigma_lin_ar.shape

ncol = int(np.ceil(np.sqrt(ntime)))
nrow = ncol
figure, ax = plt.subplots(nrow,ncol,figsize=(20,20))

# ax.plot(ibox_list)
x = svtb.ip.to_numpy()

col = np.concat([np.arange(ncol)]*nrow)
row = np.repeat(np.arange(nrow), ncol)

for itime in range(ntime):
    if itime == 0:
        ax[row[itime],col[itime]].plot(x,sigma_lin_ar[itime, 0, :], label = 'box 0', color = 'green')
        ax[row[itime],col[itime]].plot(x,sigma_lin_ar[itime, 1, :], label = 'box 1', color = 'red')
        ax[row[itime],col[itime]].plot(x,sigma_lin_ar[itime, 2, :], label = 'box 2', color = 'blue')
        ax[row[itime],col[itime]].plot(x,sigma_lin_ar[itime, 3, :], label = 'box 3', color = 'orange')
        ax[row[itime],col[itime]].plot(ip_ar[itime,0] + fip_ar[itime,0],sigma_ar[itime, 0], 'o', label = 'box 0', color = 'green')
        ax[row[itime],col[itime]].plot(ip_ar[itime,1] + fip_ar[itime,1],sigma_ar[itime, 1], 'o', label = 'box 1', color = 'red')
        ax[row[itime],col[itime]].plot(ip_ar[itime,2] + fip_ar[itime,2],sigma_ar[itime, 2], 'o', label = 'box 2', color = 'blue')
        ax[row[itime],col[itime]].plot(ip_ar[itime,3] + fip_ar[itime,3],sigma_ar[itime, 3], 'o', label = 'box 3', color = 'orange')
        ax[row[itime],col[itime]].set_ylim(-0.6,0.1)
        ax[row[itime],col[itime]].legend()
    else:
        ax[row[itime],col[itime]].plot(x,sigma_lin_ar[itime, 0, :], color = 'green')
        ax[row[itime],col[itime]].plot(x,sigma_lin_ar[itime, 1, :], color = 'red')
        ax[row[itime],col[itime]].plot(x,sigma_lin_ar[itime, 2, :], color = 'blue')
        ax[row[itime],col[itime]].plot(x,sigma_lin_ar[itime, 3, :], color = 'orange')
        ax[row[itime],col[itime]].plot(ip_ar[itime,0] + fip_ar[itime,0],sigma_ar[itime, 0], 'o', color = 'green')
        ax[row[itime],col[itime]].plot(ip_ar[itime,1] + fip_ar[itime,1],sigma_ar[itime, 1], 'o', color = 'red')
        ax[row[itime],col[itime]].plot(ip_ar[itime,2] + fip_ar[itime,2],sigma_ar[itime, 2], 'o', color = 'blue')
        ax[row[itime],col[itime]].plot(ip_ar[itime,3] + fip_ar[itime,3],sigma_ar[itime, 3], 'o', color = 'orange')
        ax[row[itime],col[itime]].set_ylim(-0.6, 0.1)

plt.tight_layout()
plt.savefig(f"sigma_{dtgw}.png")
plt.close()



figure, ax = plt.subplots(1,2)
ax[0].plot(rhs, label = 'vsim + qmodf')
ax[1].plot(rhs, label = 'vsim + qmodf')
ax[1].plot(qmodf, label = 'qmodf')
ax[1].plot(vsim_list, label = 'vsim')
ax[0].plot(lhs, label = 'sc1 * dH')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.savefig(f"wbal_{dtgw}.png")
plt.close()
