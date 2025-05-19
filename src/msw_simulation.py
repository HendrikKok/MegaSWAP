import numpy as np
from src.database import DataBase
from src.storage_formulation import StorageFormulation
from src.unsaturated_zone import UnsaturatedZone
from src.ponding import Ponding
from src.soil import Soil
from src.utils import summed_sv

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
            dtgw= 1.0
        )
        self.soil = Soil()
        self.initialize(parameters)

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

    def get_sof_parameters(self) -> tuple[float, float]:
        self.sof_conductance = self.ponding.area / self.ponding.soil_resistance
        return self.sof_conductance, self.ponding.zmax
    
    def prepare_timestep(self, itime: int, gwl:float) -> tuple[float, float]:
        self.ponding.add_precipitation(self.qrch[itime])
        self.qrch[itime] = self.ponding.get_infiltration_flux(gwl)
        self.evap_ponding = self.ponding.get_ponding_evaporation(self.pet[itime])
        if self.ponding.volume > 0.0 or gwl > self.database.mv:
            self.soil.reset()
        else:
            self.soil.update(self.qrch[itime], self.pet[itime], self.dtgw)
        self.evap_soil = self.soil.get_actual_evaporation()
        self.ds = self.unsaturated_zone.update(self.qrch[itime], gwl)
        self.vsim = self.qrch[itime] - self.ds / self.dtgw
        ig, _ = self.database.gwl_to_index(gwl)
        self.sc1 = self.get_sc1(ig, self.unsaturated_zone.ip, self.unsaturated_zone.fip)
        if gwl > self.database.mv:
            self.sc1 = 1.0
        return self.vsim, self.sc1
    
    def do_iter(self, itime: int, gwl:float) -> tuple[float, float]:
        self.ds = self.unsaturated_zone.update(self.qrch[itime], gwl)
        self.vsim = self.qrch[itime] - self.ds / self.dtgw
        ig, _ = self.database.gwl_to_index(gwl)
        self.sc1 = self.get_sc1(ig, self.unsaturated_zone.ip, self.unsaturated_zone.fip)
        if gwl > self.database.mv:
            self.sc1 = 1.0
        return self.vsim, self.sc1

    def finalise_timestep(self, gwl, qmodf, save_to_old) -> None:
        self.unsaturated_zone.finalize_timestep(gwl, qmodf, save_to_old)
        # infiltration excess based runoff
        self.qrun = self.ponding.get_runoff_flux()
        # add saturation excess runoff from mf6
        self.qrun += ((self.sof_conductance * max(0.0, gwl - self.ponding.zmax)) / self.ponding.area)
        
    def get_summed_s(self,ig,ip,fip):
        s =0.0
        for ibox in self.unsaturated_zone.non_submerged_boxes:
            s += self.database.svtb.sel(ib=ibox, ig=ig, ip=ip[ibox]).item() + fip[ibox] * (
            self.database.svtb.sel(ib=ibox, ig=ig, ip=ip[ibox] + 1).item()
            - self.database.svtb.sel(ib=ibox, ig=ig, ip=ip[ibox]).item()
        )
        return s
        
    def get_sc1(self,ig, ip, fip): 
        s1 = self.get_summed_s(ig, ip, fip)
        s2 = self.get_summed_s(ig + 1, ip, fip)
        return (s1 - s2) / (self.database.dpgwtb.loc[ig + 1] - self.database.dpgwtb.loc[ig])

    