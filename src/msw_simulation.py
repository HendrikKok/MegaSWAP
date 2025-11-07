import numpy as np
from src.database import DataBase
from src.storage_formulation import StorageFormulation, sc1_min, iterur1, iterur2, treshold, relaxation_factor
from src.unsaturated_zone import UnsaturatedZone
from src.ponding import Ponding
from src.soil import Soil
from src.utils import summed_sv
import copy

class MegaSwap:

    def __init__(self, parameters):
        self.qrch = parameters["qrch"]
        self.pet = parameters["qpet"]
        self.dtgw = parameters["dtgw"]
        self.database = DataBase(
            rootzone_dikte=parameters["rootzone_dikte"],
            mv=parameters["surface_elevation"],
            dbase_path=parameters["databse_path"],
        )
        self.storage_formulation = StorageFormulation(
            database=self.database, 
            initial_gwl=parameters["initial_gwl"],
            dtgw = self.dtgw
        )
        self.unsaturated_zone = UnsaturatedZone(
            database=self.database,
            initial_phead=parameters["initial_phead"],
            initial_gwl=parameters["initial_gwl"],
            dtgw=self.dtgw,
        )
        self.ponding = Ponding(
            zmax = 0.02, 
            area = parameters["area"],
            soil_resistance = 1.0,
            max_infiltration_rate = parameters["max_infiltration"],
            dtgw= 1.0,
            factor_ponding = parameters["factor_ponding"],

        )
        self.soil = Soil()
        self.initialize(parameters)
        self.theta = np.full(18, np.nan)

    def initialize(self, parameters: dict):
        self.gwl_mf6 = parameters["initial_gwl"]
        self.gwl_mf6_old = parameters["initial_gwl"]
        self.itime = 0
        self.storage_formulation.initialize(
            summed_sv(self.unsaturated_zone.sv),
            summed_sv(self.unsaturated_zone.sv_old),
            inital_gwl = parameters["initial_gwl"]
        )
        self.ds = 0.0  # change of storage in unsaturated zone after unsaturated_zone.update()
        self.qrun = 0.0

    def prepare_timestep(self, itime: int) -> float:
        self.itime = itime
        self.ds = self.unsaturated_zone.update(self.qrch[self.itime], self.storage_formulation.gwl_table)
        self.vsim = self.qrch[self.itime] - self.ds / self.dtgw
        self.storage_formulation.set_initial_estimate_gwl_table(self.qrch[self.itime])
        return self.vsim

    def do_iter(self, iter: int) -> float:
        self.sc1 = self.storage_formulation.update(self.gwl_mf6, iter)
        return self.sc1

    def finalise_iter(self, gwl_mf6) -> None:
        self.gwl_mf6 = gwl_mf6
        self.storage_formulation.finalise_update(self.gwl_mf6, self.qrch[self.itime], self.ds)

    def finalise_timestep(self) -> None:
        self.unsaturated_zone.finalize_timestep(self.storage_formulation.gwl_table, self.storage_formulation.qmodf)
        self.storage_formulation.finalise_timestep()


class MegaSwapExperimental(MegaSwap):
    qmax = 0.0
    evap_ponding = 0.0
    evap_soil = 0.0
    svtb_log = np.full(18, np.nan)
    sc1_bak1 = sc1_min
    iter = 0

    def get_sof_parameters(self) -> tuple[float, float]:
        self.sof_conductance = self.ponding.area / self.ponding.soil_resistance
        return self.sof_conductance, self.ponding.zmax
    
    def prepare_timestep(self, itime: int, gwl:float, qtree: float) -> tuple[float, float]:
        self.ponding.add_precipitation(self.qrch[itime])
        self.qrch[itime] = self.ponding.get_infiltration_flux(gwl)
        self.evap_ponding = 0.0
        self.evap_soil = 0.0
        if self.ponding.volume > 0.0 or gwl > self.database.mv:
            self.soil.reset()
            self.evap_ponding = self.ponding.get_ponding_evaporation(self.pet[itime])
        else:
            self.soil.update(self.qrch[itime], self.pet[itime], self.dtgw)
            self.evap_soil = self.soil.get_actual_evaporation()

        # self.qrch[itime] -= self.evap_soil
        self.qrch[itime] -= qtree
        _ , _ = self.do_iter(itime, gwl)
        return self.vsim, self.sc1
    
    def do_iter(self, itime: int, gwl:float) -> tuple[float, float]:
        self.ds = self.unsaturated_zone.update(self.qrch[itime], gwl)
        self.vsim = self.qrch[itime] - self.ds / self.dtgw
        ig, fig = self.database.gwl_to_index(gwl)
        self.sc1 = self.get_sc1(ig, self.unsaturated_zone.ip, self.unsaturated_zone.fip)
        if gwl > self.database.mv:
            self.sc1 = 1.0
        self. _save_to_stabilise_sc1()
        self.iter += 1
        return self.vsim, max(self.sc1, 0.001)
    
    def finalise_timestep(self, gwl, qmodf, save_to_old) -> None:
        self.unsaturated_zone.finalize_timestep(gwl, qmodf, save_to_old)
        # infiltration excess based runoff
        self.qrun = self.ponding.get_runoff_flux()
        # add saturation excess runoff from mf6
        self.qrun += ((self.sof_conductance * max(0.0, gwl - self.ponding.zmax)) / self.ponding.area)
        # reset iter
        self.iter = 0

    def get_thetas(self):
        ig = self.unsaturated_zone.ig_table
        fig = self.unsaturated_zone.fig_table
        ip = self.unsaturated_zone.ip
        fip = self.unsaturated_zone.fip
        self.theta[:] = np.nan
        for ibox in range(4):
            theta1d = self.database.thetatb.sel(ib=ibox, ig=ig) + fig * (self.database.thetatb.sel(ib=ibox, ig=ig + 1) - self.database.thetatb.sel(ib=ibox, ig=ig))
            ip_box = ip[ibox]
            fip_box = fip[ibox]
            self.theta[ibox] = (theta1d.sel(ip=ip_box).item() + fip_box * (theta1d.sel(ip=ip_box + 1).item() - theta1d.sel(ip=ip_box).item()))

    def get_summed_s(self,ig,ip,fip):
        s =0.0
        for ibox in self.unsaturated_zone.non_submerged_boxes:
            s += self.database.svtb.sel(ib=ibox, ig=ig, ip=ip[ibox]).item() + fip[ibox] * (
            self.database.svtb.sel(ib=ibox, ig=ig, ip=ip[ibox] + 1).item()
            - self.database.svtb.sel(ib=ibox, ig=ig, ip=ip[ibox]).item()
        )
        return s
    
    def get_box_thickness(self):
        return self.database.thickness.to_numpy()
    
    def get_svtbs(self):
        ig = self.unsaturated_zone.ig_table
        fig = self.unsaturated_zone.fig_table
        ip = self.unsaturated_zone.ip
        fip = self.unsaturated_zone.fip
        self.svtb_log[:] = np.nan
        for ibox in range(18):
            svtb1 = self.database.svtb.sel(ib=ibox, ig=ig, ip=ip[ibox]).item() + fip[ibox] * (
            self.database.svtb.sel(ib=ibox, ig=ig, ip=ip[ibox] + 1).item()
            - self.database.svtb.sel(ib=ibox, ig=ig, ip=ip[ibox]).item()
            )
            svtb2 = self.database.svtb.sel(ib=ibox, ig=ig+1, ip=ip[ibox]).item() + fip[ibox] * (
            self.database.svtb.sel(ib=ibox, ig=ig+1, ip=ip[ibox] + 1).item()
            - self.database.svtb.sel(ib=ibox, ig=ig+1, ip=ip[ibox]).item()
            )
            self.svtb_log[ibox] = (svtb1 * (1-fig)) + (svtb2 * fig) 
        return self.svtb_log
        
    def get_sc1(self,ig, ip, fip): 
        s1 = self.get_summed_s(ig, ip, fip)
        s2 = self.get_summed_s(ig + 1, ip, fip)
        return self._stabilise_sc1(((s1 - s2) / (self.database.dpgwtb.loc[ig + 1] - self.database.dpgwtb.loc[ig])).item())

    def _stabilise_sc1(self, sc1: float) -> float:
        # stabilisation in case of oscillation
        if self.iter >= iterur1 and self.iter <= iterur2:
            if (
                (self.sc1 - self.sc1_bak1) * (self.sc1_bak1 - self.sc1_bak2) < 0.0
            ):
                sc1 = sc1 * 0.5 + self.sc1_bak1 * 0.5
        omega = relaxation_factor(self.iter)
        return sc1 * omega - self.sc1_bak1 * (omega - 1.0)
    
    
    def _save_to_stabilise_sc1(self):
        self.sc1_bak2 = copy.copy(self.sc1_bak1)
        self.sc1_bak1 = copy.copy(self.sc1)