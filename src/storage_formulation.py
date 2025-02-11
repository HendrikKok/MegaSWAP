import numpy as np
import copy
from src.utils import minmax
from src.database import DataBase

sc1_min = 0.001
iterur1 = 3             # lower bound for smoothing sc1
iterur2 = 5             # uper bound for smoothing sc1
treshold = 0.00025      # treshold level for change in head in m
    
def relaxation_factor(iter: int):
    if iter <= iterur1:
        return 1.0
    elif iter > iterur2:
        return 0.0
    else:
        omegalv = 1.0 - 1.0 * (iter - iterur1) / (iterur2 - iterur1)
        return max(omegalv, 0.0)


class StorageFormulation:
    
    def __init__(self, database: DataBase, initial_gwl: float, dtgw: float):
        self.dtgw = dtgw                                                       # timestep length in days
        self.database = database                                               # class that handels all database interactions
        self.s = 0.0                                                           # integrated storage deficit unsaturated zone
        self.s_old = 0.0                                                       # integrated storage deficit unsaturated zone for t = t-1  
        self.gwl_table = copy.copy(initial_gwl)                                # groundwaterlevel based on table interpolation  
        self.gwl_table_old = copy.copy(initial_gwl)                            # groundwaterlevel based on table interpolation  for t = t-1
        self.gwl_mf6 = copy.copy(initial_gwl)                                  # groundwaterlevel from MODFLOW 6    
        self.gwl_mf6_old = copy.copy(initial_gwl)                              # groundwaterlevel from MODFLOW 6 for t = t-1 
        self.ig_mf6 = 0                                                        # storage table index for given gwl
        self.fig_mf6  = 0.0                                                    # lineair fraction between ig and ig +1, for given gwl
        self.qmodf = 0.0                                                       # contribution of MODFLOW 6 to shared water balance
        self.vcor = 0.0                                                        # correction for non convergence
        self.sc1 = sc1_min                                                     # sy for MODFLOW 6
        self.database.update_saturated_variables(self.gwl_table_old, self.gwl_mf6_old)           # set inital gwl
        self.sc1_bak1 = sc1_min
        
    def initialize(self, s, s_old, inital_gwl):
        # self.s = s
        self.s_old = s_old
        self.ig_table, self.fig_table = self.database.gwl_to_index(inital_gwl)

    def stabilise_sc1(self, gwl_mf6, sc1_balance, sc1_level, iter):
        if gwl_mf6 > self.database.mv:
            sc1 = sc1_balance
        else:            
            if self.gwl_mf6_old > self.database.mv:
                sc1 = sc1_level
            else:
                sc1 = 0.5 * (sc1_balance + sc1_level)
        # stabilisation in case of oscillation
        if iter >= iterur1 and iter <= iterur2:
            if (
                (self.sc1 - self.sc1_bak1) * (self.sc1_bak1 - self.sc1_bak2) < 0.0
            ).all():
                sc1 = sc1 * 0.5 + self.sc1_bak1 * 0.5
        omega = relaxation_factor(iter)
        return sc1 * omega - self.sc1_bak1 * (omega - 1.0)
    
    def save_to_stabilise_sc1(self):
        self.sc1_bak2 = copy.copy(self.sc1_bak1)
        self.sc1_bak1 = copy.copy(self.sc1)
        
    def get_sc1_iter1(self, gwl_mf6, gwl_table):
        self.ig_table, self.fig_table = self.database.gwl_to_index(gwl_table)
        ig_mf6, fig_mf6 = self.database.gwl_to_index(gwl_mf6)
        self.s_mf6_old = self.database.get_storage_from_gwl_index(ig_mf6, fig_mf6)
        if abs(gwl_mf6 - gwl_table) > treshold:
            # use change in storage deficit in unsaturated zone + dH of MF
            sc1 = (self.s - self.s_mf6_old) / (gwl_table - self.gwl_mf6_old)
            pass
        else:
            # no dH, use interpolated value from table
            sc1 = self.database.get_sc1_from_gwl_index(self.ig_table)
            pass
        return minmax(sc1, sc1_min, 1.0)
    
    def get_sc1(self, gwl_mf6, gwl_table):
        ig_mf6, _ = self.database.gwl_to_index(gwl_mf6)
        if abs(gwl_table - gwl_mf6) > treshold:  
            # used offset heads and change in staorage deficit
            sc1_balance = (self.s - self.s_mf6_old) / (gwl_table - self.gwl_mf6_old) 
            pass
        elif gwl_table > self.database.mv and self.gwl_mf6_old > self.database.mv:
            # ponding
            sc1_balance = 1.0
        else:
            # minimal offset, use interpolated value from table
            sc1_balance = self.database.get_sc1_from_gwl_index(self.ig_table)
        sc1_balance = minmax(sc1_balance, sc1_min, 1.0)
        if abs(gwl_mf6 - self.gwl_mf6_old) > treshold:
            sc1_level = (self.s_mf6 - self.s_mf6_old) / (gwl_mf6 - self.gwl_mf6_old)
        else:
            sc1_level = self.database.get_sc1_from_gwl_index(ig_mf6)  #TODO DEBUG!!!
        sc1_level = minmax(sc1_level, sc1_min, 1.0)
        return sc1_balance, sc1_level
    
    def set_initial_gwl_table(self, qrch):
        # initial estimate of gwl_table, based on the unsaturated zone alone
        self.s = self.s_old + qrch + self.qmodf
        self.gwl_table, self.ig_table, self.fig_table = self.database.get_gwl_table_from_storage(self.s, self.ig_table)
        pass
        
    def update(
        self,
        gwl_mf6,
        iter,
    ):
        self.vcor = 0.0
        self.save_to_stabilise_sc1()
        if iter == 1:
            sc1 = self.get_sc1_iter1(gwl_mf6, self.gwl_table)
            self.sc1 = self.stabilise_sc1(gwl_mf6, sc1, sc1, iter)
        else:
            sc1_balance, sc1_level = self.get_sc1(gwl_mf6, self.gwl_table)
            self.sc1 = self.stabilise_sc1(gwl_mf6, sc1_balance, sc1_level, iter)
        return self.sc1
    
    def finalise_update(
        self,
        gwl_mf6,
        rch_time, 
        ds,
    ):
        ig_mf6, fig_mf6 = self.database.gwl_to_index(gwl_mf6)
        self.qmodf = float((((gwl_mf6 - self.gwl_mf6_old) * self.sc1) / self.dtgw) - ((rch_time - ds) / self.dtgw))
        self.s_mf6 = self.database.get_storage_from_gwl_index(ig_mf6, fig_mf6) # storage deficit unsaturated zone based on new mf6 heads
        self.s = self.s_old  + rch_time + self.qmodf 
        self.gwl_table, self.ig_table, self.fig_table = self.database.get_gwl_table_from_storage(self.s, self.ig_table)
        self.gwl_mf6 = gwl_mf6
        return 
    
    def finalise_timestep(self):
        self.s_old = copy.copy(self.s)
        self.gwl_table_old = copy.copy(self.gwl_table)
        self.gwl_mf6_old = copy.copy(self.gwl_mf6)
        self.database.update_saturated_variables(self.gwl_table_old, self.gwl_mf6_old)




