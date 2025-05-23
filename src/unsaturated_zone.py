import numpy as np
from src.utils import phead_to_index, get_sv, init_qmv, get_qmv, summed_sv, cm2m
from src.database import DataBase
from src.utils import phead_to_index

class UnsaturatedZone:

    def __init__(
        self,
        database: DataBase,
        initial_phead: float,
        initial_gwl: float,
        dtgw: float,
    ):
        self.dtgw = dtgw                                                            # timestep length in days
        self.database = database                                                    # class that handels all database interactions
        self.phead = np.full(self.database.nbox, initial_phead, dtype = np.float32) # pressure head per box 
        self.ip, self.fip = phead_to_index(
            self.phead,
            self.database.ddpptb,
            self.database.nuip,
        )                                                                           # storage table index and linear fraction for phead and ig + 1
        self.ig_table, self.fig_table  = self.database.gwl_to_index(
            initial_gwl
        )                                                                           # lineair fraction between ig and ig +1, for given gwl
        self.sv = np.zeros(self.database.nbox, dtype = np.float32)                  # storage deficit per box
        self.sv_old = np.zeros(self.database.nbox, dtype = np.float32)              # storage deficit per box, for t -1
        self.qmv = np.zeros(self.database.nbox, dtype = np.float32)                 # bottom flux per box
        for ibox in range(self.database.nbox):
            self.sv_old[ibox] = get_sv(
                self.ig_table, self.fig_table, self.ip[ibox], self.fip[ibox], self.database.svtb, ibox
            )
        self.sv = np.copy(self.sv_old)
        self.qmv = np.zeros(self.database.nbox)
        for ibox in range(self.database.nbox):
            self.qmv[ibox] = init_qmv(self.ig_table, self.fig_table, self.ip[ibox], self.fip[ibox],self.database.qmrtb)
        self.qmv_old = np.copy(self.qmv)    

    def _get_qrch(self, qrch, gwl_table) -> float:
        if gwl_table >= 1000:  #self.database.mv
            ig, _ = self.database.gwl_to_index(gwl_table)
            ip, _ = phead_to_index(self.phead, self.database.ddpptb, self.database.nuip)
            ip = ip[0]
            q = -self.database.qmrtb.sel(ig=ig,ip=ip).item()
            return min(qrch, q)
        else:
            return qrch

    def update(self, qrch, gwl_table, new_time:bool=False):
        self.qmv_old = np.copy(self.qmv)
        # update phead with default values: equilibrium pressure head 
        # for the current groundwater level
        dpgw = self.database.box_top[0] - gwl_table
        self.phead[:] = -dpgw + 0.5 * self.database.rootzone_dikte
        # updates unsaturated zone for fixed gwl and given recharge on top
        self.non_submerged_boxes = self.database.get_non_submerged_boxes(gwl_table)
        # qrch = self._get_qrch(qrch,gwl_table)
        if new_time:
            self.qmf = np.copy(self.qmv_old)
        for ibox in self.non_submerged_boxes:
            #TODO: should we use the self.database.get_max_ig
            if ibox == 0:
                qin = -qrch
            else:
                qin = self.qmv[ibox - 1]
            sigma = self.sv_old[ibox] - qin * self.dtgw
            self.ip[ibox], self.fip[ibox] = self.database.sigma2ip(
                sigma, self.ig_table, self.fig_table, ibox, self.dtgw, self.ip[ibox], self.fip[ibox]
            )
            if self.ip[ibox] < 0:
                self.phead[ibox] = self.database.ptb["value"][self.ip[ibox]] + self.fip[
                    ibox
                ] * (
                    self.database.ptb["value"][self.ip[ibox] + 1]
                    - self.database.ptb["value"][self.ip[ibox]]
                )
            else:
                self.phead[ibox] = -cm2m*(10**((self.ip[ibox] + self.fip[ibox])*self.database.ddpptb))
            self.sv[ibox] = get_sv(
                self.ig_table, self.fig_table, self.ip[ibox], self.fip[ibox], self.database.svtb, ibox
            )
            self.qmv[ibox] = get_qmv(
                self.sv[ibox], self.sv_old[ibox], ibox, qin, self.qmv, self.dtgw
            )
        self.database.update_unsaturated_variables(self.qmv, self.phead, self.non_submerged_boxes)
        return summed_sv(self.sv) - summed_sv(self.sv_old)

    def finalize_timestep(self, gwl_table, qmodf: float, save_to_old: bool = True):
        # updates the internal states of unsaturated zone, based on the new gwl
        self.ig_table, self.fig_table = self.database.gwl_to_index(
            gwl_table
        )  
        self.non_submerged_boxes = self.database.get_non_submerged_boxes(gwl_table)
        ibmax = self.database.nbox - 1  # 0-based
        # updates storage deficit
        for ibox in range(ibmax):
            self.sv[ibox] = get_sv(
                self.ig_table, self.fig_table, self.ip[ibox], self.fip[ibox], self.database.svtb, ibox
            )
        # update qmv's
        self.qmv[:] = 0.0
        self.qmv[ibmax - 1] = -(self.sv[ibmax] - self.sv_old[ibmax]) / self.dtgw + qmodf
        for ibox in range(ibmax - 2, -1, -1):
            self.qmv[ibox] = (
                -(self.sv[ibox + 1] - self.sv_old[ibox + 1]) / self.dtgw
                + self.qmv[ibox + 1]
            )
        self.qmv[ibmax] = (self.sv[ibmax] - self.sv_old[ibmax]) / self.dtgw + self.qmv[
            ibmax - 1
        ]
        # update prz
        for ibox in self.non_submerged_boxes:
            self.ip[ibox], self.fip[ibox] = self.database.sv2ip(
                self.sv[ibox], self.ig_table, self.fig_table, ibox, self.dtgw, self.ip[ibox], self.fip[ibox]
            )
            if self.ip[ibox] < 0:
                self.phead[ibox] = self.database.ptb["value"][self.ip[ibox]] + self.fip[
                    ibox
                ] * (
                    self.database.ptb["value"][self.ip[ibox] + 1]
                    - self.database.ptb["value"][self.ip[ibox]]
                )
            else:
                self.phead[ibox] = -cm2m*(10**((self.ip[ibox]+self.fip[ibox])*self.database.ddpptb))
        # save to old
        if save_to_old:
            self.sv_old = np.copy(self.sv)
        return
