import numpy as np
import xarray as xr
import pandas as pd
from src.utils import cm2m, dc, phead_to_index, qmr2ip


class DataBase:

    def __init__(self, rootzone_dikte: float, mv: float, dbase_path: str):
        tabel = xr.open_dataset(dbase_path)
        tabel["svtb"] = tabel["svtb"].fillna(0.0)                          # needed for fill value issues in nc
        tabel["qmrtb"] = tabel["qmrtb"].fillna(0.0)                        # needed for fill value issues in nc
        self.mv = mv                                                       # surface level 
        self.rootzone_dikte = rootzone_dikte                               # rootzone thickness   
        self.set_constants_from_tabel(tabel)                               # set all needed variables from nc
        self.set_arrays(tabel)
        self.storage_tabel = np.full(self.svtb.ig.shape, np.nan)           # array to store temp-storage deficite estimates 
        
    def set_constants_from_tabel(self, tabel: xr.Dataset) -> None:
        self.ddpptb = tabel.ddpptb
        self.ddgwtb = tabel.ddgwtb
        self.nuip = tabel.nuip
        self.nuig = tabel.nuig
        self.nlig = tabel.nlig
        self.nlip = tabel.nlip
        self.nxlig = tabel.nxlig
        self.nxuig = tabel.nuig
        self.dpczsl = tabel.dpczsl
        self.nbox = 18
        self.box_bottom = np.array(
            [
                self.mv - self.rootzone_dikte,
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
            data={"index_gwtb": tabel.igdc.to_numpy().astype(dtype=np.int32)},
            index=np.arange(tabel.igdcmn, tabel.igdcmx + 1, 1),
        )
        self.dpgwtb = pd.DataFrame(
            data={"value": tabel["dpgwtb"].fillna(0.0).to_numpy()},
            index=np.arange(self.nxlig - 1, self.nxuig + 1, 1),
        )
        ig = np.arange(self.nxlig - 1, self.nxuig + 1, 1)
        ip = np.arange(self.nlip, self.nuip + 1, 1)
        ib = np.arange(0, 18, 1)
        self.svtb = tabel["svtb"].assign_coords(
            {
                "ip": ip,
                "ig": ig,
                "ib": ib,
            }
        )
        self.qmrtb = tabel["qmrtb"].assign_coords(
            {
                "ip": ip,
                "ig": ig,
            }
        )
        ptb_index = np.arange(self.nlip, self.nuip + 1)
        ptb_values = np.zeros_like(ptb_index, dtype=np.float64)
        ptb_values[ptb_index <= 0] = -self.ddpptb * ptb_index[ptb_index <= 0]
        ptb_values[ptb_index > 0] = -cm2m * (
            10 ** (ptb_index[ptb_index > 0] * self.ddpptb)
        )
        self.ptb = pd.DataFrame(data={"value": ptb_values}, index=ptb_index)

    def ip_in_bounds(self, ip, fip):
        if ip > self.svtb.ip.values[-2]:
            ip = np.array(self.svtb.ip.values[-2])
            fip = np.ones(1, dtype=np.float64)
            print("out of bounds phead")
        return ip, fip

    def get_ig_box_bottom(self) -> np.ndarray:
        # perched conditions?
        # TODO: 0.15 from database dpczsl
        ig_box_bottom = np.full_like(self.box_bottom, -999, dtype=np.int32)
        ig_index = np.arange(1, self.dpgwtb.index[-1] + 1, 1)
        lower = -self.dpgwtb["value"][ig_index - 1].to_numpy()
        upper = -self.dpgwtb["value"][ig_index].to_numpy()
        for bottom, index in zip(self.box_bottom, range(self.box_bottom.size)):
            ig_box_bottom[index] = ig_index[(lower > bottom) & (upper <= bottom)]
        return ig_box_bottom.astype(dtype=np.int32)
    
    def get_max_ig(self, ig, ib, ip) -> int:
        ig_box_bottom = self.get_ig_box_bottom()
        igmax = ig
        if ig_box_bottom[ib] < ig:
            for igtp in range(ig, ig_box_bottom[ib], -1):
                if self.qmrtb.sel(ig=igtp, ip=ip) < 0.0:
                    if self.qmrtb.sel(ig=igtp, ip=ip) < self.qmrtb.sel(ig=igmax, ip=ip):
                        igmax = igtp
        return igmax
    
    def sigma2ip(self, sigma, ig, fig, ibox: int, dtgw: float) -> tuple[int, float]:
        sigmabtb = self.svtb.sel(ib=ibox) - dtgw * self.qmrtb
        sigma1d = sigmabtb.sel(ig=ig) + fig * (
            sigmabtb.sel(ig=ig + 1) - sigmabtb.sel(ig=ig)
        )
        if np.unique(sigma1d).size == 1:
            return None, None
        sorter = np.argsort(sigma1d)
        sorted_index = np.searchsorted(sigma1d, sigma, sorter=sorter)
        if sorted_index >= sorter.size:
            # ip_index = sigmabtb.ip.max().item()
            ip_index = sorter[sorted_index - 1].item()
        else:
            ip_index = sorter[sorted_index].item()
        # if ip_index >= sigmabtb.ip.max():
        #     print("out of max bounds..")
        #     ip = ip_index - 1
        #     fip = 1.0
        # elif ip_index <= sigmabtb.ip.min():
        #     ip = sigmabtb.ip.min()
        #     fip = 0
        # else:
        ip = sigmabtb.ip[ip_index].item()
        if ip == sigmabtb.ip.min():
            fip = 0.0
        elif ip >= sigmabtb.ip.max():
            ip = ip -1
            fip = 1.0
        else:
            fip = (
                (sigma - sigma1d[ip_index])
                / (sigma1d[ip_index + 1] - sigma1d[ip_index])
            ).item()
        return ip, fip

    def sv2ip(self, sv, ig, fig, ibox: int, dtgw: float) -> tuple[int, float]:
        sigmabtb = self.svtb.sel(ib=ibox) # - dtgw * self.qmrtb
        sigma1d = sigmabtb.sel(ig=ig) + fig * (
            sigmabtb.sel(ig=ig + 1) - sigmabtb.sel(ig=ig)
        )
        if np.unique(sigma1d).size == 1:
            return None, None
        sorter = np.argsort(sigma1d)
        sorted_index = np.searchsorted(sigma1d, sv, sorter=sorter)
        if sorted_index >= sorter.size:
            ip_index = sigmabtb.ip.max()
        else:
            ip_index = sorter[sorted_index].item()
        if ip_index >= sigmabtb.ip.max():
            print("out of max bounds..")
            ip = sigmabtb.ip.max().item() - 1
            fip = 1.0
        elif ip_index < sigmabtb.ip.min():
            ip = sigmabtb.ip.min()
            fip = 0
        else:
            ip = sigmabtb.ip[ip_index].item()
            fip = (
                (sv - sigma1d[ip_index]) / (sigma1d[ip_index + 1] - sigma1d[ip_index])
            ).item()
        return ip, fip
    
    def gwl_to_index(self, gwl) -> tuple[np.ndarray, np.ndarray]:
        dpgw = self.mv - gwl
        igk = self.igdc["index_gwtb"][
            np.floor((dpgw + dc) / self.ddgwtb)
        ] 
        # ponding
        if dpgw < 0.0:
            igk -= 1
        # maximize on array size?
        igk = np.maximum(igk, self.nlig - 1)
        figk = (dpgw - self.dpgwtb["value"][igk]) / (
            self.dpgwtb["value"][igk + 1] - self.dpgwtb["value"][igk]
        )
        return igk, figk

    def get_non_submerged_boxes(self, gwl):
        ig, _ = self.gwl_to_index(gwl)
        dpgw = np.ones(self.box_top.size) * self.dpgwtb["value"].loc[ig].item()
        mask =  dpgw >= -self.box_top
        mask[0] = True
        return np.arange(self.box_top.size)[mask]
        
    def get_phead_index_estimate(self, ig, ibox, peq, peq_table):
        iprz, fiprz = phead_to_index(self.phead[ibox], self.ddpptb, self.nuip)
        if ig > self.ig_table_old and self.phead[ibox] > peq_table and self.dpgw_table_old < -self.box_bottom[ibox]:
            if self.dpgw_table_old < 0.05 or self.phead[ibox] < peq:
                # CASE 1: set to equilibrium value; from cap-rise to percolation
                pnew = peq_table
                ip, fip = phead_to_index(pnew, self.ddpptb, self.nuip)
                return self.ip_in_bounds(ip, fip)
            elif self.phead[ibox] > peq:
                # CASE 2: semi-implicit scheme; from percolation to percolation
                ip, fip = self.ip_in_bounds(iprz, fiprz)
                qmvtp = self.qmrtb.sel(ig=ig, ip=ip).item() + fip * (
                    self.qmrtb.sel(ig=ig, ip=ip + 1).item()
                    - self.qmrtb.sel(ig=ig, ip=ip).item()
                )
                # average; allow for an increase of the percolation
                if qmvtp < 0.0 and qmvtp < self.qmv[ibox]:
                    qmvtp = qmvtp * 0.5 + self.qmv[ibox] * 0.5
                # from qmvtp find phead and indexes
                ip, fip = qmr2ip(qmvtp, ig, self.svtb.ip, self.qmrtb)
                ip, fip = self.ip_in_bounds(ip, fip)
                if ip < 0:
                    pnewtp = self.ptb["value"][ip] + fip * (
                        self.ptb["value"][ip + 1] - self.ptb["value"][ip]
                    )
                else:
                    pnewtp = -cm2m*(10**((ip+fip)*self.ddpptb))
                pnew = self.phead[ibox] - (self.dpgwtb["value"][ig] - self.dpgw_table_old)
                if pnew > pnewtp:
                    # update index based on maximum dragdown
                    ip, fip = phead_to_index(pnew, self.ddpptb, self.nuip)
                    return self.ip_in_bounds(ip, fip)
                else:
                    return self.ip_in_bounds(iprz, fiprz)
            else:
                # CASE 0: default: use indexes based phead of unsaturated zone alone
                return self.ip_in_bounds(iprz, fiprz)
        else:
            # CASE 0: default: use indexes based phead of unsaturated zone alone
            return self.ip_in_bounds(iprz, fiprz)

    def create_storage_tabel_boxes(
        self, ig
    ):
        sgwln = 0.0
        for ibox in self.non_submerged_boxes:
            sgwln += self.create_storage_tabel_box(
                ibox, ig
            )
        return sgwln

    def create_storage_tabel_box(self, ibox, ig):
        #  equilibrium pressure head for the current groundwater level
        peq = (
            -self.dpgw_table_old + 0.5 * self.rootzone_dikte
        ) # current situation ('from')
        
        peq_table = (
            -self.dpgwtb["value"][ig] + 0.5 * self.rootzone_dikte
        )  # new situation ('to')
        
        ip, fip = self.get_phead_index_estimate(ig, ibox, peq, peq_table)
        return self.svtb.sel(ib=ibox, ig=ig, ip=ip).item() + fip * (
            self.svtb.sel(ib=ibox, ig=ig, ip=ip + 1).item()
            - self.svtb.sel(ib=ibox, ig=ig, ip=ip).item()
        )
        
    def get_storage_from_gwl_index(self, ig, fig):
        self.storage_tabel[ig] = self.create_storage_tabel_boxes(ig)
        self.storage_tabel[ig + 1] = self.create_storage_tabel_boxes(ig + 1)
        return self.storage_tabel[ig] + fig * (
            self.storage_tabel[ig + 1] - self.storage_tabel[ig]
        )
    
    def get_gwl_table_from_storage(self, sarg, ig_start):
        ig = ig_start
        self.storage_tabel[ig] = self.create_storage_tabel_boxes(ig)
        self.storage_tabel[ig + 1] = self.create_storage_tabel_boxes(ig + 1)
        dif = self.storage_tabel[ig + 1] - sarg
        ig = None
        if dif < 0.0:
            for ig in range(ig_start, -1, -1):
                self.storage_tabel[ig] = self.create_storage_tabel_boxes(ig)
                if (
                    sarg <= self.storage_tabel[ig]
                    and sarg >= self.storage_tabel[ig + 1]
                ):
                    break
        else:
            for ig in range(ig_start, 51, 1):
                self.storage_tabel[ig + 1] = self.create_storage_tabel_boxes(ig)
                if (
                    sarg <= self.storage_tabel[ig]
                    and sarg >= self.storage_tabel[ig + 1]
                ):
                    break
        fig = (sarg - self.storage_tabel[ig]) / (
            self.storage_tabel[ig + 1]
            - self.storage_tabel[ig]
        )
        return (
            self.mv
            - (
                self.dpgwtb.loc[ig]
                + fig
                * (self.dpgwtb.loc[ig + 1] - self.dpgwtb.loc[ig])
            )
        )[0].item(), ig, fig
        
    def get_sc1_from_gwl_index(self, ig):
        self.storage_tabel[ig] = self.create_storage_tabel_boxes(ig)
        self.storage_tabel[ig + 1] = self.create_storage_tabel_boxes(ig + 1)
        return (
            self.storage_tabel[ig] - self.storage_tabel[ig + 1]
        ) / (
            self.dpgwtb.loc[ig + 1]
            - self.dpgwtb.loc[ig]
        )
        
    def update_unsaturated_variables(self, qmv, phead, non_submerged_boxes):
        self.qmv = qmv
        self.phead = phead
        self.non_submerged_boxes = non_submerged_boxes
        
    def update_saturated_variables(self, gwl_table_old, gwl_mf6_old):
        self.dpgw_table_old = self.mv - gwl_table_old
        self.ig_table_old, _ = self.gwl_to_index(gwl_table_old)