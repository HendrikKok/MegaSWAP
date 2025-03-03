import numpy as np
from src.database import DataBase
from src.storage_formulation import StorageFormulation
from src.unsaturated_zone import UnsaturatedZone
from src.utils import summed_sv

class MegaSwap:

    def __init__(self, parameters):
        self.qrch = parameters["qrch"]
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

    def prepare_timestep(self, itime: int) -> float:
        self.itime = itime
        self.ds = self.unsaturated_zone.update(self.qrch[self.itime], self.storage_formulation.gwl_table)
        self.vsim = self.qrch[self.itime] - self.ds / self.dtgw
        self.storage_formulation.set_initial_gwl_table(self.qrch[self.itime])
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
    
    def prepare_timestep(self, itime: int, gwl:float) -> tuple[float, float]:
        self.ds = self.unsaturated_zone.update(self.qrch[itime], gwl)
        self.vsim = self.qrch[itime] - self.ds / self.dtgw
        ig, _ = self.database.gwl_to_index(gwl)
        self.sc1 = self.get_sc1(ig, self.unsaturated_zone.ip, self.unsaturated_zone.fip)
        return self.vsim, self.sc1
    
    def finalise_timestep(self, gwl, qmodf, save_to_old) -> None:
        # self.unsaturated_zone.sv_old = np.copy(self.unsaturated_zone.sv)
        self.unsaturated_zone.finalize_timestep(gwl, qmodf, save_to_old)
        
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

    