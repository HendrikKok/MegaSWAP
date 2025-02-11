import numpy as np
from src.database import DataBase
from src.storage_formulation import StorageFormulation
from src.unsaturated_zone import UnsaturatedZone

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
            self.unsaturated_zone.sv.sum(),
            self.unsaturated_zone.sv_old.sum(),
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

