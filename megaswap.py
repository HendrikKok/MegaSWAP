import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

m2cm = 100.0
cm2m = 1/100.0

def pf2head(pf: np.ndarray) -> np.ndarray:
    return -10**(pf/10)

def head2pf(phead: np.ndarray) -> np.ndarray:
    return np.log10(-m2cm*phead)

def sigma2ip(sigma, ig, fig, ibox:int, svtb:xr.DataArray, qmrtb:xr.DataArray, dtgw: float) -> tuple[int,float]:
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
    return ip, fip


def sv2ip(sv, ig, fig, ibox:int, svtb:xr.DataArray, qmrtb:xr.DataArray, dtgw: float) -> tuple[int,float]:
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
    return ip, fip

def qmr2ip(qmr, ig, sigmabtb: xr.DataArray, qmrtb: xr.DataArray) -> tuple[int,float]:
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

def get_q(ip, fip, ig, fig, qmrtb: xr.DataArray) -> np.ndarray:
    qmr1d = (qmrtb.sel(ig=ig) + fig * (qmrtb.sel(ig=ig+1) - qmrtb.sel(ig=ig))).to_numpy().ravel()
    return qmr1d[ip] + fip * (qmr1d[ip + 1] - qmr1d[ip])

def get_qmv_bd(svnew: np.ndarray, svold: np.ndarray, dtgw: float) -> np.ndarray:
    qmv = np.zeros_like(svnew)
    nxb = qmv.size
    qmv[nxb + 1] = -(svnew[nxb]-svold[nxb]) / dtgw
    for ibox in range(nxb-2,0,-1):
        qmv[ibox] = -(svnew[ibox + 1]-svold[ibox + 1]) / dtgw + qmv[ibox + 1]
    return qmv

def init_qmv(ig, fig, ip, fip, qmrtb: xr.DataArray):
    qmv_ip1 = qmrtb.sel(ig=ig,ip=ip) + fig * (qmrtb.sel(ig=ig+1,ip=ip) - qmrtb.sel(ig=ig,ip=ip))
    qmv_ip2 = qmrtb.sel(ig=ig,ip=ip + 1) + fig * (qmrtb.sel(ig=ig+1,ip=ip+1) - qmrtb.sel(ig=ig,ip=ip+1))
    return qmv_ip1 + fip * (qmv_ip2 - qmv_ip1)

def get_sv(ig, fig, ip, fip, svtb: xr.DataArray, ib=None):
    if ib is None:
        ib_range = slice(svtb.ib[0],svtb.ib[-1])
    else:
        ib_range = ib
    sv_lin = svtb.sel(ig=ig,ib = ib_range) + fig * (svtb.sel(ig=ig+1,ib = ib_range) - svtb.sel(ig=ig,ib = ib_range))
    return (sv_lin.sel(ip=ip) + fip * (sv_lin.sel(ip=ip+1) - sv_lin.sel(ip=ip))).to_numpy().ravel().item()

def get_qmv(svnew: np.ndarray, svold: np.ndarray, ibox,qin, qmv_in, dtgw: float) -> np.ndarray:
    qmv = np.copy(qmv_in)
    if ibox == 0:
        # return (svnew - svold) / dtgw - qin
        return qin + (svnew - svold) / dtgw
    else:
        return qmv[ibox - 1] + (svnew - svold) / dtgw
    
def summed_sv(sv):
    s = 0.0
    for ibox in sv.size-1:
        s +=sv[ibox]
    return s

def phead_to_index(phead: np.ndarray, ddpptb:float, nuip: int) -> tuple[np.ndarray,np.ndarray]:
    if not isinstance(phead, np.ndarray):
        phead = np.array([phead])
    # positive ph = ponding -> so linear
    pFtb = -phead/ddpptb
    ip = pFtb.astype(dtype= np.int32) - 1  # int function for <0 rounds towards zero
    fip  = pFtb - ip
    # ph from zero to -1 cm 
    mask = phead < 0.0
    ip[mask] = 0
    fip[mask] = 0
    # ph from -1 cm onwards
    mask = phead < -1 * cm2m
    pFtb[mask] = np.log10(-m2cm*phead[mask])/ddpptb
    ip[mask] = np.minimum(pFtb[mask].astype(dtype=np.int32), nuip - 1)
    fip[mask] = pFtb[mask] - ip[mask]
    # min and maximize fractions
    fip = np.maximum(fip, 0.0)
    fip = np.minimum(fip, 1.0)
    return ip, fip


class DataBase:
    
    def __init__(self, # The code `rootzone_dikte` is not performing any specific action in the
    # provided snippet. It seems to be a variable name or identifier in Python
    # code, but without any context or usage, it is difficult to determine its
    # purpose or functionality.
    rootzone_dikte: float, mv: float, dbase_path: str):
        tabel = xr.open_dataset(dbase_path)
        tabel['svtb'] = tabel['svtb'].fillna(0.0)
        tabel['qmrtb'] = tabel['qmrtb'].fillna(0.0)
        self.set_arrays(tabel)
        self.mv = mv
        self.rootzone_dikte = rootzone_dikte
        self.set_constants_from_tabel(tabel)
    
    def set_constants_from_tabel(self, tabel: xr.Dataset) -> None:
        self.ddpptb = tabel.ddpptb
        self.ddgwtb = tabel.ddgwtb
        self.nuip = tabel.nuip
        self.nuig = tabel.nuig
        self.nlig   = tabel.nlig
        self.nlip = tabel.nlip
        self.nxlig = tabel.nxlig
        self.nxuig = tabel.nuig
        self.dpczsl = tabel.dpczsl
        self.nbox = 18
        self.box_bottom = np.array(
            [self.mv-self.rootzone_dikte,
            -1.0 - self.dpczsl,
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
        self.box_top = np.zeros_like(self.mv)
        self.box_top[1:] = self.box_bottom[1:] - np.diff(self.box_bottom)
        
    def set_arrays(self, tabel) -> None:
        self.igdc = pd.DataFrame(
            data = {'index_gwtb': tabel.igdc.to_numpy().astype(dtype=np.int32)
            }, index = np.arange(tabel.igdcmn, tabel.igdcmx + 1, 1)
        )
        self.dpgwtb = pd.DataFrame(
            data = {'value': tabel['dpgwtb'].fillna(0.0).to_numpy()
            }, index = np.arange(self.nxlig - 1 , self.nxuig + 1, 1)
        )
        ig = np.arange(self.nxlig - 1 , self.nxuig + 1, 1)
        ip = np.arange(self.nlip , self.nuip + 1, 1)
        ib = np.arange(0,18,1)
        self.svtb = tabel['svtb'].assign_coords({
            'ip': ip,
            'ig': ig,
            'ib': ib,
        })
        self.qmrtb = tabel['qmrtb'].assign_coords({
            'ip': ip,
            'ig': ig,
        })
        ptb_index = np.arange(self.nlip, self.nuip + 1)
        ptb_values = np.zeros_like(ptb_index, dtype=np.float64)
        ptb_values[ptb_index <= 0] = -self.ddpptb * ptb_index[ptb_index <= 0]
        ptb_values[ptb_index > 0] = -cm2m*(10**(ptb_index[ptb_index > 0] * self.ddpptb))
        self.ptb = pd.DataFrame(
            data = {'value': ptb_values},
            index = ptb_index
        )

    def create_storage_tabel_boxes(self, ig, ig_old, prz, lvgw_old, non_submerged_boxes): 
        dpgwold = self.mv - lvgw_old
        sgwln= 0.0 
        for ibox in non_submerged_boxes:
            sgwln += self.create_storage_tabel_box(ibox, ig, ig_old, prz[ibox], dpgwold)
        return sgwln.item()

    def create_storage_tabel_box(self, ibox, ig, igold, prz, dpgwold, dtgw:float, qmv):
        sigmabtb = self.svtb.sel(ib = ibox) - dtgw * self.qmrtb
        iprz, fiprz = phead_to_index(prz, self.ddpptb, self.nuip)
        # current situation ('from') 
        peq    = -dpgwold + 0.5 * self.rootzone_dikte  #  equilibrium pressure head for the currentn groundwater level
        # new situation ('to')
        peqtb  = -self.dpgwtb['value'][ig] + 0.5 * self.rootzone_dikte  #  presure head from position in the dpgwtb-table.
        # only for deeper table positions ig than the current one igkold
        # only if the current situation does not have plocking
        # only if the 'to' situation is percolation
        # only if the current groundwater level is above the bottom of the box
        if ig > igold and prz > peqtb:
            if dpgwold < 0.05 or prz < peq:
                # set to equilibrium value; from cap-rise to percolation 
                pnew = peqtb
                ip, fip = phead_to_index(pnew, self.ddpptb, self.nuip)
            elif prz > peq:
                # semi-implicit scheme; from percolation to percolation
                ip = iprz  
                fip = fiprz
                qmvtp = self.qmrtb.sel(ig=ig,ip=ip).item() + fip * (self.qmrtb.sel(ig = ig, ip=ip+1).item() - self.qmrtb.sel(ig=ig, ip= ip).item())
                # average ; allow for an increase of the percolation
                if qmvtp < 0.0 and qmvtp < qmv[ibox]:
                    qmvtp = qmvtp*0.5 + qmv[ibox]*0.5
                # from qmvtp find phead and indexes
                ip, fip = qmr2ip(qmvtp, ig, sigmabtb,self.qmrtb)
                pnewtp = self.ptb['value'][ip] + fip * (self.ptb['value'][ip + 1] - self.ptb['value'][ip])
                pnew  = prz - (self.dpgwtb['value'][ig] - dpgwold)
                if pnew > pnewtp:
                    # update index based on maximum dragdown 
                    ip, fip = phead_to_index(pnew, self.ddpptb, self.nuip)
                else:
                    ip = iprz
                    fip = fiprz
            else:
                ip = iprz
                fip = fiprz
        else:
            ip = iprz
            fip = fiprz
        return self.svtb.sel(ib = ibox, ig=ig, ip=ip).item() + fip * (self.svtb.sel(ib = ibox, ig=ig, ip=ip + 1).item()  - self.svtb.sel(ib = ibox, ig=ig, ip=ip).item())
    
    def get_ig_box_bottom(self) -> np.ndarray:
        # perched conditions? 
        # TODO: 0.15 from database dpczsl
        # box_bottom[1] = box_bottom[1] - tabel.dpczsl
        ig_box_bottom = np.full_like(self.box_bottom,-999,dtype = np.int32)
        ig_index = np.arange(1,self.dpgwtb.index[-1]+1,1)
        lower = -self.dpgwtb['value'][ig_index-1].to_numpy()
        upper = -self.dpgwtb['value'][ig_index].to_numpy()
        for bottom,index in zip(self.box_bottom,range(self.box_bottom.size)):
            ig_box_bottom[index] = ig_index[(lower > bottom) & (upper <= bottom)]
        return ig_box_bottom.astype(dtype=np.int32)

    # ig_slrzbox -> ig-index of bottoms
    def get_max_ig(self,ig,ib,ip) -> int:
        ig_box_bottom = self.get_ig_box_bottom()
        igmax = ig
        if ig_box_bottom[ib] < ig:
            for igtp in range(ig,ig_box_bottom[ib],-1):
                if self.qmrtb.sel(ig=igtp,ip=ip) < 0.0:
                    if self.qmrtb.sel(ig=igtp,ip=ip) < self.qmrtb.sel(ig=igmax,ip=ip):
                        igmax = igtp
        return igmax


class StorageFormulation:
    
    def __init__(self, database: DataBase, sc1min: float):
        self.database = database
        self.storage_tabel = np.zeros(database.svtb.ig.shape)
        self.sc1min = sc1min
        
    def add_storage_tabel_element
        
        
    def update(self, lvgw, lvgw_old, s, sold, prz, ig, ig_old, non_submerged_boxes) -> any:
        ig_mf6 = ig
        treshold = 0.00025
        self.storage_tabel[ig] = self.database.create_storage_tabel_boxes(ig, ig_old, prz, lvgw_old, non_submerged_boxes)
        if abs(lvgw - lvgw_old) > treshold:
            # sc1 waterbalance 
            if lvgw > mv or lvgw_old > self.database.mv:
                sc1_wb = (s - sold) / (lvgw - lvgw_old)
            else:
                self.storage_tabel[ig] = self.database.create_storage_tabel_boxes(ig, ig_old, prz, lvgw_old,non_submerged_boxes)
                self.storage_tabel[ig + 1] = self.database.create_storage_tabel_boxes(ig + 1, ig_old, prz, lvgw_old,non_submerged_boxes)
                sc1_wb = (self.storage_tabel[ig] - self.storage_tabel[ig + 1])/ (self.database.dpgwtb.loc[ig + 1] - self.database.dpgwtb.loc[ig])
            sc1_wb = np.maximum(sc1_wb,self.sc1min)
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
            sc1 = np.maximum(sc1, self.sc1min)
        elif lvgw > self.database.mv:
            sc1 = 1.0
        else:
            # no change, use trajectory value
            self.storage_tabel[ig_mf6] = self.database.create_storage_tabel_boxes(ig_mf6, ig_old, prz, lvgw_old, non_submerged_boxes)
            self.storage_tabel[ig_mf6 + 1] = self.database.create_storage_tabel_boxes(ig_mf6 + 1, ig_old, prz, lvgw_old, non_submerged_boxes)
            sc1 = (self.storage_tabel[ig_mf6] - self.storage_tabel[ig_mf6 + 1])/(self.database.dpgwtb.loc[ig_mf6+1] - self.database.dpgwtb.loc[ig_mf6])
        sc1 = np.maximum(sc1, self.sc1min)
        sc1 = np.minimum(sc1,1.0)
        return sc1, self.storage_tabel
    
    def finalise(self)-> None:
        self.storage_tabel[:] = 0.0

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
        
class UnsaturatedZone:
    dc = 0.1e-6
    
    def __init__(self,database: DataBase, storage_formulation: StorageFormulation, dtgw:float):
        self.database = database
        self.storage_formulation = storage_formulation
        self.dtgw = dtgw
        self.sv = np.zeros(self.database.nbox)
        self.qmv = np.zeros(self.database.nbox)
        self.sv_old = np.zeros_like(self.sv)
        self.storage_tabel = np.zeros(self.database.nbox)
        self.phead = np.zeros(self.database.nbox)

    def update(self, ig, fig, qrch):
        self.qmv[:] = 0.0
        ip, fip  = phead_to_index(self.phead, self.database.ddpptb, self.database.nuip)
        non_submerged_boxes = self.get_non_submerged_boxes(self.database.box_top, ig)
        for ibox in non_submerged_boxes:
            ig_local = self.database.get_max_ig(ig, ibox, ip[ibox]) 
            if ig != ig_local:
                pass
            # add recharge to get new volume
            if ibox == 0:
                qin = -qrch
            else:
                qin = self.qmv[ibox - 1]
            sigma = self.sv_old[ibox] - qin * self.dtgw 
            ip[ibox], fip[ibox] = sigma2ip(sigma, ig_local, fig, ibox)
            self.phead[ibox] = self.database.ptb['value'][ip[ibox]] + fip[ibox] * (self.database.ptb['value'][ip[ibox] + 1] - self.database.ptb['value'][ip[ibox]])
            self.sv[ibox] = get_sv(ig_local,fig,ip[ibox],fip[ibox],self.database.svtb, ibox)
            self.qmv[ibox] = get_qmv(self.sv[ibox], self.sv_old[ibox], ibox, qin, self.qmv)
        return summed_sv(self.sv), summed_sv(self.sv_old), self.phead
    
    def get_gwl(self, sarg, prz, lvgw_old, ig_start, ig_old):
        ig = ig_start
        self.storage_formulation.storage_tabel[ig] = self.database.create_storage_tabel_boxes(ig, ig_old, prz, lvgw_old)
        self.storage_formulation.storage_tabel[ig + 1] = self.database.create_storage_tabel_boxes(ig, ig_old, prz, lvgw_old)
        dif = self.storage_formulation.storage_tabel[ig+1] - sarg
        ig = None
        if dif < 0.0:
            for ig in range(ig_start, -1, -1):
                self.storage_formulation.storage_tabel[ig] = self.database.create_storage_tabel_boxes(ig, ig_old, prz, lvgw_old)
                if sarg <= self.storage_formulation.storage_tabel[ig] and sarg >= self.storage_formulation.storage_tabel[ig+1]:
                    break
        else:
            for ig in range(ig_start, 51, 1):
                self.storage_formulation.storage_tabel[ig + 1] = self.database.create_storage_tabel_boxes(ig + 1, ig_old, prz, lvgw_old)
                if sarg <= self.storage_formulation.storage_tabel[ig] and sarg >= self.storage_formulation.storage_tabel[ig+1]:
                    break
        fig = (sarg-self.storage_formulation.storage_tabel[ig])/(self.storage_formulation.storage_tabel[ig+1]-self.storage_formulation.storage_tabel[ig])
        return self.database.mv - (self.database.dpgwtb.loc[ig] + fig*(self.database.dpgwtb.loc[ig+1]-self.database.dpgwtb.loc[ig])), ig, fig

    def finalize(self, ig,fig,ip,fip,qmodf):
        # based on msw1bd
        # update sv based on new pressure head
        ibmax = self.database.nbox - 1 # 0-based
        #TODO: fix issue with new sv's
        for ibox in range(ibmax):
            ig_local = get_max_ig(ig, ibox, ip[ibox], ig_box_bottom) 
            sv[ibox] = get_sv(ig_local,fig,ip[ibox],fip[ibox],ibox)
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
            phead[ibox] = self.database.ptb['value'][ip[ibox]] + fip[ibox] * (self.database.ptb['value'][ip[ibox] + 1] - self.database.ptb['value'][ip[ibox]])
        return sv, qmv, ip, fip, phead
    
    def get_non_submerged_boxes(self, gwl):
        ig, _ = self.gwl_to_index(gwl)
        mask = self.database.dpgwtb['value'].loc[ig] >= -self.database.box_top
        return np.arange(self.database.box_top.size)[mask]

    def gwl_to_index(self, gwl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        below_bot = gwl < (self.database.mv - self.database.box_bottom)
        gwl[below_bot] = self.database.mv - self.database.box_bottom
        dpgw = self.database.mv - gwl
        igk = self.database.igdc['index_gwtb'][np.floor((dpgw + self.dc) / self.database.box_bottom.ddgwtb)].to_numpy()  # ddgwtb = stepsize in igdc array
        # ponding
        ponding = dpgw < 0.0
        igk[ponding] = igk[ponding] - 1
        # maximize on array size?
        igk = np.maximum(igk, self.database.nlig - 1)
        figk = (dpgw - self.database.dpgwtb['value'][igk].to_numpy()) / (self.database.dpgwtb['value'][igk + 1].to_numpy() - self.database.dpgwtb['value'][igk].to_numpy())
        return igk.item(),figk.item()

class MegaSwap:
    s: np.ndarray
    def __init__(self, parameters):
        self.database = DataBase(
            parameters["rootzone_dikte"],
            mv = parameters["surface_elevation"],
            dbase_path = parameters["databse_path"],
            )
        self.unsaturated_zone = UnsaturatedZone(database=self.database)
        self.storage_formulation = StorageFormulation(database=self.database, sc1min=0.001)
        self.init()
        self.qrch = parameters["qrch"]
        self.dtgw = self.database["dtgw"]
        
    def init(self):
        self.gwl_old = parameters["initial_gwl"]
        self.gwl = parameters["initial_gwl"]
        self.phead_old = np.full(self.database.nbox, parameters["initial_phead"])
        
        
    def prepare_timestep(self, itime: int) -> float:
        ig, fig = self.unsaturated_zone.gwl_to_index(self.gwl)
        self.s, self.s_old, self.phead = self.unsaturated_zone.update(ig, fig, self.qrch[itime])
        self.vsim = qrch - (self.s - self.s_old) / self.dtgw
        return self.vsim
    
    def do_iter(self, gwl) -> float:
        non_submerged_boxes = self.unsaturated_zone.get_non_submerged_boxes(gwl)
        self.sc1, self.storage_tabel = self.storage_formulation.update(gwl, self.gwl_old, self.s, self.s_old, self.phead, self.storage_tabel, non_submerged_boxes)
        self.gwl_old = np.copy(gwl)
        self.gwl_msw = self.unsaturated_zone.get_gwl(sarg, sgwln, self.phead, self.gwl_old, ig_start, self.ig_old)
        return self.sc1
    self.gwl_old
    def finalise_iter(self) -> None:
        self.storage_formulation.finalise()
    
    def finalise_timestep(self, gwl) -> None:
        qmodf = (self.sc1 * (self.gwl_old - gwl) - self.vsim)
        ig, fig = self.unsaturated_zone.gwl_to_index(gwl)
        self.sv, self.qmv, ip, fip, self.phead = self.unsaturated_zone.finalize(ig,fig, self.ip, self.fip, qmodf)
        self.save_to_old(gwl,ig,ip,fip)
        
    def save_to_old(self, gwl, ig, ip,fip) -> None:
        self.gwl_old = np.copy(gwl)
        self.phead_old = np.copy(self.phead)
        self.svold = np.copy(self.sv)
        self.ig_old = np.copy(ig)
        self.ip_old, self.fip_old = np.copy(ip), np.copy(fip)

## entry point of script ##
qrch = np.array([0.0016]*160)
parameters = {
    "databse_path": 'database\\unsa_300.nc',
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "surface_elevation": 0.0,
    "initial_gwl": -6.9,
    "initial_phead": -(0 - -6.9),
    "dtgw": 1.0,
}

megaswap = MegaSwap(parameters)

# stand alone MegaSwap run
ntime = qrch.size
niter = 1

phead_log = np.zeros((ntime + 1, 18))
phead_log[0,:] = parameters['initial_phead']
gwl_log = np.zeros((ntime + 1))
gwl_log[0] = parameters["initial_gwl"]

for itime in range(ntime):
    vsim = megaswap.prepare_timestep(itime)
    for iter in range(niter):
        gwl = None
        sc1 = megaswap.do_iter(gwl)
        gwl_log[itime + 1] = megaswap.gwl_msw
    megaswap.
    megaswap.finalise_timestep(gwl_log[itime + 1])
    phead_log[itime + 1] = np.copy(megaswap.phead)  # logging
    

