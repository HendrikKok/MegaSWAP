import numpy as np
from src.database import DataBase
from src.utils import phead_to_index


class Ponding:
    """ 
    keeps track of ponding volumes
    """
    def __init__(
            self, 
            zmax, 
            area, 
            soil_resistance,
            max_infiltration_rate,
            dtgw,
            factor_ponding = 1.0,
            ):
            self.zmax = zmax
            self.area = area
            self.volume = 0.0
            self.stage = 0.0
            self.soil_resistance = soil_resistance
            self.max_infiltration_rate = max_infiltration_rate
            self.dtgw = dtgw
            self.factor_ponding = factor_ponding

    def _set_stage(self) -> None:
         self.stage = self.volume / self.area

    def _add_volume(self, volume):
        self.volume += volume
        self.volume  = max(0.0, self.volume) 

    def add_precipitation(self, pp) -> None:
        self._add_volume(pp * self.area * self.dtgw)  # net precipitation
        self._set_stage()

    def get_runoff_flux(self) -> float:
        # rejected infiltration based runoff
        qrunoff = max(0.0, self.stage - self.zmax)
        self._add_volume(-(qrunoff * self.area * self.dtgw))
        self._set_stage()
        return qrunoff
    
    def get_infiltration_flux(self, gwl) -> float:
        # no unsaturated zone, remove all stored water
        if gwl > self.zmax:
            qinf = (self.volume / self.area) / self.dtgw
            self.volume = 0.0
        else:
            qinf = min(self.max_infiltration_rate, (self.volume / self.area) / self.dtgw)
            self._add_volume(-(qinf * self.area * self.dtgw))
        self._set_stage()
        return qinf 
    
    def get_ponding_evaporation(self, pet):
        if self.stage > 0.0:
            qpt = pet * self.factor_ponding * self.dtgw
            self._add_volume(-(qpt * self.area))
        else:
            qpt = 0.0
        return qpt
