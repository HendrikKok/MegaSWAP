import numpy as np
import xarray as xr
import pandas as pd
import copy

m2cm = 100.0
cm2m = 1 / 100.0
dc = 0.1e-6
sc1_min = 0.001
iter_bnd1 = 3
iter_bnd2 = 5

def minmax(v, vmin, vmax) -> float:
    return np.minimum(np.maximum(v, vmin), vmax)

def pf2head(pf: np.ndarray) -> np.ndarray:
    return -(10 ** (pf / 10))

def head2pf(phead: np.ndarray) -> np.ndarray:
    return np.log10(-m2cm * phead)

def qmr2ip(qmr, ig, ip_indexes: xr.DataArray, qmrtb: xr.DataArray) -> tuple[int, float]:
    qmr1d = qmrtb.sel(ig=ig)
    sorter = np.argsort(qmr1d)
    ip_qmr1d = sorter[np.searchsorted(qmr1d, qmr, sorter=sorter)].item()
    ip = qmr1d.ip[sorter[ip_qmr1d]].item()
    if ip >= ip_indexes.max():
        print("out of bounds..")
        ip = ip_indexes.max().item()
        fip = 0
        return ip, fip
    fip = (qmr - qmr1d.sel(ip=ip).item()) / (
        qmr1d.sel(ip=ip+1).item() - qmr1d.sel(ip=ip).item()
    )
    if fip < 0.0: # bacause of sorter action
        ip -= 1 
        fip = (qmr - qmr1d.sel(ip=ip).item()) / (
            qmr1d.sel(ip=ip+1).item() - qmr1d.sel(ip=ip).item()
        )
    return ip, fip

def get_q(ip, fip, ig, fig, qmrtb: xr.DataArray) -> np.ndarray:
    qmr1d = (
        (qmrtb.sel(ig=ig) + fig * (qmrtb.sel(ig=ig + 1) - qmrtb.sel(ig=ig)))
        .to_numpy()
        .ravel()
    )
    return qmr1d[ip] + fip * (qmr1d[ip + 1] - qmr1d[ip])

def get_qmv_bd(svnew: np.ndarray, svold: np.ndarray, dtgw: float) -> np.ndarray:
    qmv = np.zeros_like(svnew)
    nxb = qmv.size
    qmv[nxb + 1] = -(svnew[nxb] - svold[nxb]) / dtgw
    for ibox in range(nxb - 2, 0, -1):
        qmv[ibox] = -(svnew[ibox + 1] - svold[ibox + 1]) / dtgw + qmv[ibox + 1]
    return qmv

def init_qmv(ig, fig, ip, fip, qmrtb: xr.DataArray):
    qmv_ip1 = qmrtb.sel(ig=ig, ip=ip) + fig * (
        qmrtb.sel(ig=ig + 1, ip=ip) - qmrtb.sel(ig=ig, ip=ip)
    )
    qmv_ip2 = qmrtb.sel(ig=ig, ip=ip + 1) + fig * (
        qmrtb.sel(ig=ig + 1, ip=ip + 1) - qmrtb.sel(ig=ig, ip=ip + 1)
    )
    return qmv_ip1 + fip * (qmv_ip2 - qmv_ip1)

def get_sv(ig, fig, ip, fip, svtb: xr.DataArray, ib):
    if ib is None:
        ib_range = slice(svtb.ib[0], svtb.ib[-1])
    else:
        ib_range = ib
    # intp over ig
    sv_lin = svtb.sel(ig=ig, ib=ib_range) + fig * (
        svtb.sel(ig=ig + 1, ib=ib_range) - svtb.sel(ig=ig, ib=ib_range)
    )
    # dan over ip
    return (
        (sv_lin.sel(ip=ip) + fip * (sv_lin.sel(ip=ip + 1) - sv_lin.sel(ip=ip)))
        .to_numpy()
        .ravel()
        .item()
    )

def get_qmv(
    svnew: np.ndarray, svold: np.ndarray, ibox, qin, qmv_in, dtgw: float
) -> np.ndarray:
    qmv = np.copy(qmv_in)
    if ibox == 0:
        return qin + (svnew - svold) / dtgw
    else:
        return qmv[ibox - 1] + (svnew - svold) / dtgw

def summed_sv(sv):
    s = 0.0
    for ibox in range(sv.size - 1):
        s += sv[ibox]
    return s

def phead_to_index(
    phead: np.ndarray, ddpptb: float, nuip: int
) -> tuple[np.ndarray, np.ndarray]:
    # if not isinstance(phead, np.ndarray):
    #     phead = np.array([phead])
    # # positive ph = ponding -> so linear   np.floor((dpgw + dc) / self.ddgwtb)
    # ph from zero to -1 cm
    def phead_per_item(phead):
        if phead > 0.0:
            ip = 0
            fip = 0.0
        elif phead < -1 * cm2m:
            pFtb = np.log10(-m2cm * phead) / ddpptb
            ip = int(np.minimum(np.floor(pFtb), nuip - 1))
            fip = pFtb - float(ip)
            return ip, fip
        else:
            pFtb = -phead / ddpptb
            ip = int(np.floor(pFtb) - 1)  # int function for <0 rounds towards zero 
            fip = pFtb - float(ip)
        return ip, fip
    
    if isinstance(phead, np.ndarray):
        ip = np.zeros(phead.shape, dtype = np.int32)
        fip = np.zeros(phead.shape, dtype = np.float64)
        for i in range(phead.size):
            ip[i], fip[i] = phead_per_item(phead[i])
    else:
        ip, fip = phead_per_item(phead)
    return ip, fip


