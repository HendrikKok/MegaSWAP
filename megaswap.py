import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import copy

m2cm = 100.0
cm2m = 1/100.0
dc = 0.1e-6
sc1_min = 0.001
iter_bnd1 = 4
iter_bnd2 = 6

def minmax(v, vmin, vmax) -> float:
    return np.minimum(np.maximum(v,vmin), vmax)

def pf2head(pf: np.ndarray) -> np.ndarray:
    return -10**(pf/10)

def head2pf(phead: np.ndarray) -> np.ndarray:
    return np.log10(-m2cm*phead)

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

def get_sv(ig, fig, ip, fip, svtb: xr.DataArray, ib):
    if ib is None:
        ib_range = slice(svtb.ib[0],svtb.ib[-1])
    else:
        ib_range = ib
    sv_lin = svtb.sel(ig=ig,ib = ib_range) + fig * (svtb.sel(ig=ig+1,ib = ib_range) - svtb.sel(ig=ig,ib = ib_range))
    return (sv_lin.sel(ip=ip) + fip * (sv_lin.sel(ip=ip+1) - sv_lin.sel(ip=ip))).to_numpy().ravel().item()

def get_qmv(svnew: np.ndarray, svold: np.ndarray, ibox,qin, qmv_in, dtgw: float) -> np.ndarray:
    qmv = np.copy(qmv_in)
    if ibox == 0:
        return qin + (svnew - svold) / dtgw
    else:
        return qmv[ibox - 1] + (svnew - svold) / dtgw
    
def summed_sv(sv):
    s = 0.0
    for ibox in range(sv.size-1):
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
    
    def __init__(self, rootzone_dikte: float, mv: float, dbase_path: str):
        tabel = xr.open_dataset(dbase_path)
        tabel['svtb'] = tabel['svtb'].fillna(0.0)
        tabel['qmrtb'] = tabel['qmrtb'].fillna(0.0)
        self.mv = mv
        self.rootzone_dikte = rootzone_dikte
        self.set_constants_from_tabel(tabel)
        self.set_arrays(tabel)
        
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
        self.box_top = np.zeros_like(self.box_bottom)
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

    def create_storage_tabel_boxes(self, ig, ig_old, prz, lvgw_old, non_submerged_boxes, qmv, dtgw): 
        dpgwold = self.mv - lvgw_old
        sgwln= 0.0 
        for ibox in non_submerged_boxes:
            sgwln += self.create_storage_tabel_box(ibox, ig, ig_old, prz[ibox], dpgwold, qmv, dtgw)
        return sgwln.item()

    def create_storage_tabel_box(self, ibox, ig, igold, prz, dpgwold, qmv, dtgw:float):
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
                ip, fip = self.ip_in_bounds(ip, fip)
            elif prz > peq:
                # semi-implicit scheme; from percolation to percolation
                ip, fip = self.ip_in_bounds(iprz, fiprz)
                qmvtp = self.qmrtb.sel(ig=ig,ip=ip).item() + fip * (self.qmrtb.sel(ig = ig, ip=ip+1).item() - self.qmrtb.sel(ig=ig, ip= ip).item())
                # average ; allow for an increase of the percolation
                if qmvtp < 0.0 and qmvtp < qmv[ibox]:
                    qmvtp = qmvtp*0.5 + qmv[ibox]*0.5
                # from qmvtp find phead and indexes
                ip, fip = qmr2ip(qmvtp, ig, sigmabtb,self.qmrtb)
                ip, fip = self.ip_in_bounds(ip, fip)
                pnewtp = self.ptb['value'][ip] + fip * (self.ptb['value'][ip + 1] - self.ptb['value'][ip])
                pnew  = prz - (self.dpgwtb['value'][ig] - dpgwold)
                if pnew > pnewtp:
                    # update index based on maximum dragdown 
                    ip, fip = phead_to_index(pnew, self.ddpptb, self.nuip)
                    ip, fip = self.ip_in_bounds(ip, fip)
                else:
                    ip, fip = self.ip_in_bounds(iprz, fiprz)
            else:
                ip, fip = self.ip_in_bounds(iprz, fiprz)
        else:
            ip, fip = self.ip_in_bounds(iprz, fiprz)
        return self.svtb.sel(ib = ibox, ig=ig, ip=ip).item() + fip * (self.svtb.sel(ib = ibox, ig=ig, ip=ip + 1).item()  - self.svtb.sel(ib = ibox, ig=ig, ip=ip).item())
    
    def ip_in_bounds(self, ip, fip):
        if ip > self.svtb.ip.values[-2]:
            ip = np.array(self.svtb.ip.values[-2])
            fip = np.ones(1, dtype=np.float64)
            print('out of bounds phead')
        return ip, fip
    
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
    
    def sigma2ip(self, sigma, ig, fig, ibox:int, dtgw: float) -> tuple[int,float]:
        sigmabtb = self.svtb.sel(ib = ibox) - dtgw * self.qmrtb  
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
    
    def sv2ip(self, sv, ig, fig, ibox:int, dtgw: float) -> tuple[int,float]:
        sigmabtb = self.svtb.sel(ib = ibox) - dtgw * self.qmrtb  
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
    
    def gwl_to_index(self, gwl) -> tuple[np.ndarray, np.ndarray]:
        dpgw = self.mv - gwl
        igk = self.igdc['index_gwtb'][np.floor((dpgw + dc) / self.ddgwtb)]  # ddgwtb = stepsize in igdc array
        # ponding
        if dpgw < 0.0:
            igk -= 1
        # maximize on array size?
        igk = np.maximum(igk, self.nlig - 1)
        figk = (dpgw - self.dpgwtb['value'][igk]) / (self.dpgwtb['value'][igk + 1] - self.dpgwtb['value'][igk])
        return igk,figk
    
    def get_non_submerged_boxes(self, gwl):
        ig, _ = self.gwl_to_index(gwl)
        mask = self.dpgwtb['value'].loc[ig] >= -self.box_top
        return np.arange(self.box_top.size)[mask]

class StorageFormulation:
    
    def __init__(self, database: DataBase, sc1min: float):
        self.database = database
        self.storage_tabel = np.full(database.svtb.ig.shape, np.nan)
        self.sc1min = sc1min
        self.sc1_bak1 = sc1_min
        self.sc1_bak2 = sc1_min
        self.sc1 = sc1_min
        
    def add_storage_tabel_element(self, ig_st, ig, ig_old, prz, lvgw_old, non_submerged_boxes, qmv, dtgw) -> None:
        if np.isnan(self.storage_tabel[ig_st]):
            self.storage_tabel[ig_st] = self.database.create_storage_tabel_boxes(ig, ig_old, prz, lvgw_old, non_submerged_boxes, qmv, dtgw)
        
    def update(self, gwl_table, gwl_table_old, gwl_mf6, gwl_mf6_old, s, s_old, s_mf6,s_mf6_old, prz, non_submerged_boxes, qmv, dtgw, iter, ig_mf6, fig_mf6):
        treshold = 0.00025
        ig_table, _ = self.database.gwl_to_index(gwl_table)
        ig_table_old, _ = self.database.gwl_to_index(gwl_table_old)

        self.sc1_bak2 = copy.copy(self.sc1_bak1)
        self.sc1_bak1 = copy.copy(self.sc1)
        itype = np.nan
        if iter == 0:
            if abs(gwl_mf6 -gwl_mf6_old) > treshold:
                # use change in storage deficit in unsaturated zone + dH of MF 
                sc1 = (s - s_old) / (gwl_mf6 - gwl_mf6_old)
                itype = 1.0
            else:
                # no dH, use interpolated value from table
                self.add_storage_tabel_element(ig_table, ig_table, ig_table_old, prz, gwl_table_old, non_submerged_boxes, qmv, dtgw)
                self.add_storage_tabel_element(ig_table + 1, ig_table + 1, ig_table_old, prz, gwl_table_old, non_submerged_boxes, qmv, dtgw)
                sc1 = (self.storage_tabel[ig_table] - self.storage_tabel[ig_table + 1])/ (self.database.dpgwtb.loc[ig_table + 1] - self.database.dpgwtb.loc[ig_table])
                itype = 2.0
            sc1 = minmax(sc1, sc1_min, 1.0)
            sc1_level = sc1
            sc1_balance = sc1
        else:
            if abs(gwl_table - gwl_mf6) > treshold:
                # used offset heads and change in staorage deficit
                sc1_balance = (s - s_mf6_old) / (gwl_table - gwl_mf6_old)
                itype = 3.0
            elif gwl_table > self.database.mv and gwl_mf6_old > self.database.mv:
                # ponding
                sc1_balance = 1.0
                itype = 4.0
            else:
                # minimal offset, use interpolated value from table
                self.add_storage_tabel_element(ig_table, ig_table, ig_table_old, prz, gwl_table_old, non_submerged_boxes, qmv, dtgw)
                self.add_storage_tabel_element(ig_table + 1, ig_table + 1, ig_table_old, prz, gwl_table_old, non_submerged_boxes, qmv, dtgw)
                sc1_balance = (self.storage_tabel[ig_table] - self.storage_tabel[ig_table + 1])/ (self.database.dpgwtb.loc[ig_table + 1] - self.database.dpgwtb.loc[ig_table])
                itype = 5.0
            sc1_balance = minmax(sc1_balance, sc1_min, 1.0)
            
            if abs(gwl_mf6 - gwl_mf6_old) > treshold:
                sc1_level = (s_mf6 - s_mf6_old) / (gwl_mf6 - gwl_mf6_old)
            else:
                self.add_storage_tabel_element(ig_mf6, ig_mf6, ig_table_old, prz, gwl_table_old, non_submerged_boxes, qmv, dtgw)
                self.add_storage_tabel_element(ig_mf6 + 1, ig_mf6 + 1, ig_table_old, prz, gwl_table_old, non_submerged_boxes, qmv, dtgw)
                sc1_level = (self.storage_tabel[ig_mf6] - self.storage_tabel[ig_mf6 + 1])/ (self.database.dpgwtb.loc[ig_mf6 + 1] - self.database.dpgwtb.loc[ig_mf6])
            sc1_level = minmax(sc1_level, sc1_min, 1.0)
        if gwl_mf6 > self.database.mv:
            self.sc1 = sc1_balance
        else:
            # from ponding
            if gwl_mf6_old > self.database.mv:
                self.sc1 = sc1_level
            else:
                self.sc1 = 0.5 * (sc1_balance + sc1_level)
                
        if iter >= 3 and iter <=iter_bnd1:
            if ((self.sc1 - self.sc1_bak1) * (self.sc1_bak1 - self.sc1_bak2) < 0.0).all():
                # stabilisation in case of oscillation
                self.sc1 = self.sc1 * 0.5 + self.sc1_bak1 * 0.5
        # omega = self.relaxation_factor(iter)
        # self.sc1 = self.sc1 * omega - self.sc1_bak1 * (omega - 1.0)
        return self.sc1, itype
    
    def prepare_update(self, gwl_mf6, gwl_mf6_old, s_old, ds, dtgw, prz, gwl_table, gwl_table_old, non_submerged_boxes, qmv):
        ig_mf6, fig_mf6 = self.database.gwl_to_index(gwl_mf6)
        ig_table_old, _ = self.database.gwl_to_index(gwl_table)
        # storage deficit unsaturated zone based on new mf6 heads
        self.add_storage_tabel_element(ig_mf6, ig_mf6, ig_table_old, prz, gwl_table_old, non_submerged_boxes, qmv, dtgw)
        self.add_storage_tabel_element(ig_mf6 + 1, ig_mf6 + 1, ig_table_old, prz, gwl_table_old, non_submerged_boxes, qmv, dtgw)
        
        s_mf6 = self.storage_tabel[ig_mf6] + fig_mf6 * (self.storage_tabel[ig_mf6 + 1] - self.storage_tabel[ig_mf6])
        qmodf = (gwl_mf6 - gwl_mf6_old) * self.sc1 - ds * dtgw
        s = s_old + qmodf
        return s_mf6, s, ig_mf6, fig_mf6, qmodf

    def finalise(self)-> None:
        self.storage_tabel[:] = np.nan

    def relaxation_factor(self, iter:int):
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
    
    def __init__(self,database: DataBase, storage_formulation: StorageFormulation, dtgw:float, initial_phead:float, initial_gwl:float):
        self.database = database
        self.storage_formulation = storage_formulation
        self.dtgw = dtgw
        self.qmv = np.zeros(self.database.nbox)
        self.storage_tabel = np.zeros(self.database.nbox)
        self.phead = np.full(self.database.nbox, initial_phead)
        self.init_soil(initial_phead, initial_gwl)
        
    def init_soil(self, initial_phead: float, initial_gwl: float) -> None:
        self.sv_old = np.zeros(self.database.nbox)
        self.ip, self.fip = phead_to_index(np.full(self.database.nbox, initial_phead), self.database.ddpptb, self.database.nuip)
        ig, fig = self.database.gwl_to_index(initial_gwl)
        for ibox in range(self.database.nbox):
            self.sv_old[ibox] = get_sv(ig,fig,self.ip[ibox],self.fip[ibox],self.database.svtb,ibox)  
        self.sv = np.copy(self.sv_old)

    def update(self, ig, fig, qrch, gwl):
        self.qmv[:] = 0.0
        ip, _  = phead_to_index(self.phead, self.database.ddpptb, self.database.nuip)
        non_submerged_boxes = self.database.get_non_submerged_boxes(gwl)
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
            self.ip[ibox], self.fip[ibox] = self.database.sigma2ip(sigma, ig_local, fig, ibox, self.dtgw)
            self.phead[ibox] = self.database.ptb['value'][self.ip[ibox]] + self.fip[ibox] * (self.database.ptb['value'][self.ip[ibox] + 1] - self.database.ptb['value'][self.ip[ibox]])
            self.sv[ibox] = get_sv(ig_local,fig,self.ip[ibox],self.fip[ibox],self.database.svtb, ibox)
            self.qmv[ibox] = get_qmv(self.sv[ibox], self.sv_old[ibox], ibox, qin, self.qmv, self.dtgw)
        return summed_sv(self.sv), summed_sv(self.sv_old), self.phead
    
    def get_gwl_table(self, sarg, prz, lvgw_old, ig_start, ig_old, non_submerged_boxes):
        ig = ig_start
        self.storage_formulation.add_storage_tabel_element(ig, ig, ig_old, prz, lvgw_old,non_submerged_boxes, self.qmv, self.dtgw)
        self.storage_formulation.add_storage_tabel_element(ig + 1, ig, ig_old, prz, lvgw_old,non_submerged_boxes, self.qmv, self.dtgw)
        dif = self.storage_formulation.storage_tabel[ig+1] - sarg
        ig = None
        if dif < 0.0:
            for ig in range(ig_start, -1, -1):
                self.storage_formulation.add_storage_tabel_element(ig, ig, ig_old, prz, lvgw_old,non_submerged_boxes, self.qmv, self.dtgw)
                if sarg <= self.storage_formulation.storage_tabel[ig] and sarg >= self.storage_formulation.storage_tabel[ig+1]:
                    break
        else:
            for ig in range(ig_start, 51, 1):
                self.storage_formulation.add_storage_tabel_element(ig + 1, ig + 1, ig_old, prz, lvgw_old,non_submerged_boxes, self.qmv, self.dtgw)
                if sarg <= self.storage_formulation.storage_tabel[ig] and sarg >= self.storage_formulation.storage_tabel[ig+1]:
                    break
        fig = (sarg-self.storage_formulation.storage_tabel[ig])/(self.storage_formulation.storage_tabel[ig+1]-self.storage_formulation.storage_tabel[ig])
        return (self.database.mv - (self.database.dpgwtb.loc[ig] + fig*(self.database.dpgwtb.loc[ig+1]-self.database.dpgwtb.loc[ig])))[0].item()

    def finalize(self, ig, fig, qmodf, gwl):
        # based on msw1bd
        # update sv based on new pressure head
        ibmax = self.database.nbox - 1 # 0-based
        for ibox in range(ibmax):
            ig_local = self.database.get_max_ig(ig, ibox, self.ip[ibox]) 
            self.sv[ibox] = get_sv(ig_local,fig,self.ip[ibox],self.fip[ibox],self.database.svtb, ibox)
        # update qmv's
        self.qmv[:] = 0.0
        self.phead[:] = 0.0
        self.qmv[ibmax - 1] = -(self.sv[ibmax] - self.sv_old[ibmax]) / self.dtgw + qmodf
        if self.qmv[0] > 0.0:
            raise ValueError('inflow box 1 from bottom')
        for ibox in range(ibmax - 2,-1,-1):
            self.qmv[ibox] = -(self.sv[ibox + 1] - self.sv_old[ibox + 1]) / self.dtgw + self.qmv[ibox+1]
        self.qmv[ibmax] = (self.sv[ibmax] - self.sv_old[ibmax]) / self.dtgw + self.qmv[ibmax - 1]
        # update prz
        for ibox in self.database.get_non_submerged_boxes(gwl):  
            self.ip[ibox], self.fip[ibox] = self.database.sv2ip(self.sv[ibox], ig_local, fig, ibox, self.dtgw)
            self.phead[ibox] = self.database.ptb['value'][self.ip[ibox]] + self.fip[ibox] * (self.database.ptb['value'][self.ip[ibox] + 1] - self.database.ptb['value'][self.ip[ibox]])
        return self.ip, self.fip, self.phead
    
    def save_to_old(self) -> None:
        self.sv_old = np.copy(self.sv)

class MegaSwap:

    def __init__(self, parameters):
        self.qrch = parameters["qrch"]
        self.dtgw = parameters["dtgw"]
        self.database = DataBase(
            parameters["rootzone_dikte"],
            mv = parameters["surface_elevation"],
            dbase_path = parameters["databse_path"],
            )
        self.storage_formulation = StorageFormulation(
            database=self.database, 
            sc1min=0.001
        )
        self.unsaturated_zone = UnsaturatedZone(
            database = self.database, 
            storage_formulation = self.storage_formulation,
            dtgw = self.dtgw,
            initial_phead = parameters["initial_phead"],
            initial_gwl = parameters["initial_gwl"]
        )
        self.initialize(parameters)
        
    def initialize(self, parameters: dict):
        self.gwl_table_old = parameters["initial_gwl"]
        self.gwl_table = parameters["initial_gwl"]
        self.gwl_mf6 = parameters["initial_gwl"]
        self.gwl_mf6_old = parameters["initial_gwl"]
        self.ig_table_old = None
        self.s_mf6_old = None
        
    def prepare_timestep(self, itime: int) -> float:
        ig, fig = self.database.gwl_to_index(self.gwl_table)
        self.s, self.s_old, self.phead = self.unsaturated_zone.update(ig, fig, self.qrch[itime], self.gwl_table)
        self.vsim = self.qrch[itime] - (self.s - self.s_old) / self.dtgw
        if self.ig_table_old is None:
            self.ig_table_old = np.copy(ig)
        return self.vsim
    
    def do_iter(self, gwl_mf6: float, iter: int) -> float:
        self.ig, _ = self.unsaturated_zone.database.gwl_to_index(self.gwl_table)
        ig_mf6,fig_mf6 = self.unsaturated_zone.database.gwl_to_index(self.gwl_mf6)
        
        self.non_submerged_boxes = self.database.get_non_submerged_boxes(self.gwl_table) 
        self.ds = (self.s - self.s_old) # change in storage deficit due to initial update unsaturated zone
        #self.s_mf6 = copy.copy(self.s)
        #s = self.s
        self.s_mf6, s, ig_mf6, fig_mf6, self.qmodf = self.storage_formulation.prepare_update(
            gwl_mf6, 
            self.gwl_mf6_old, 
            self.s_old, 
            self.ds, 
            self.dtgw, 
            self.phead, 
            self.gwl_table,
            self.gwl_table_old,
            self.non_submerged_boxes,
            self.unsaturated_zone.qmv,
            )
        if self.s_mf6_old is None:
            self.s_mf6_old = copy.copy(self.s_mf6)
        self.gwl_table = self.unsaturated_zone.get_gwl_table(self.s_mf6,self.phead,self.gwl_table_old,self.ig, self.ig_table_old, self.non_submerged_boxes)
        self.sc1, sf_type = self.storage_formulation.update(
            self.gwl_table, 
            self.gwl_table_old, 
            gwl_mf6, 
            self.gwl_mf6_old, 
            s, # new update 
            self.s_old, 
            self.s_mf6,
            self.s_mf6_old, 
            self.phead, 
            self.non_submerged_boxes, 
            self.unsaturated_zone.qmv, 
            self.unsaturated_zone.dtgw, 
            iter,
            ig_mf6,
            fig_mf6
        )
        return self.sc1, self.non_submerged_boxes.size, sf_type
    
    def finalise_iter(self) -> None:
        self.storage_formulation.finalise()
    
    def finalise_timestep(self, gwl) -> float:        
        ig, fig = self.database.gwl_to_index(self.gwl_table)
        ip, fip, self.phead = self.unsaturated_zone.finalize(ig, fig, self.qmodf, gwl)
        self.storage_formulation.finalise()
        self.save_to_old(gwl,ig,ip,fip)
        return self.qmodf
        
    def save_to_old(self, gwl, ig, ip, fip) -> None:
        self.gwl_table_old = np.copy(gwl)
        self.ig_table_old = np.copy(ig)
        self.ip_old, self.fip_old = np.copy(ip), np.copy(fip)
        self.s_mf6_old = copy.copy(self.s_mf6)
        self.s_old = np.copy(self.s)
        self.gwl_mf6_old = copy.copy(self.gwl_mf6)
        self.unsaturated_zone.save_to_old()
        
    def get_gwl(self) -> float:
        self.non_submerged_boxes = self.database.get_non_submerged_boxes(self.gwl_table)
        ig, _ = self.unsaturated_zone.database.gwl_to_index(self.gwl_table)
        return self.unsaturated_zone.get_gwl_table(self.s,self.phead,self.gwl_table_old,ig,self.ig_table_old, self.non_submerged_boxes)
