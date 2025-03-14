import math
import copy


class Soil:
    """ 
    keeps track of Booesten and Stroosnijder bare soil evaporation
    """
    def __init__(self, soil_parameter: float = 0.054):
        self.boosten_soil_parameter = soil_parameter
        self.net_pp = 0.0
        self.pot_evaporation = 0.0
        self.cum_pot_evaporation = 0.0
        self.cum_act_evaporation = 0.0
        self.cum_act_evaporation_old = 0.0
        self.dt = 1.0

    def update(self, net_precipitation, potential_evaporation, dt):
        self.net_pp = net_precipitation
        self.pot_evaporation = potential_evaporation
        self.dt = dt
        self._update_cumulative_vars() 

    def reset(self):
        # in case of ponding dont use (pp < et) as startingpoint of dry period
        self.cum_act_evaporation = 0.0
        self.cum_act_evaporation_old = 0.0
        self.cum_pot_evaporation = 0.0

    def _update_cumulative_vars(self):
        self.cum_act_evaporation_old = copy.copy(self.cum_act_evaporation)
        # add deficit to potential soil evaporation
        self.cum_pot_evaporation += max(0.0, (self.pot_evaporation - self.net_pp) * self.dt)
        # derive actual soil evaporation
        if self.net_pp < self.pot_evaporation:
            if self.cum_pot_evaporation <= self.boosten_soil_parameter**2:
                self.cum_act_evaporation = self.cum_pot_evaporation
            else:
                self.cum_act_evaporation = self.boosten_soil_parameter * math.sqrt(self.cum_pot_evaporation)
        else:
            self.cum_act_evaporation -= (self.net_pp - self.pot_evaporation) * self.dt
            self.cum_act_evaporation = max(0.0, self.cum_act_evaporation) # restart in case of new drying period
            if self.cum_act_evaporation < self.boosten_soil_parameter**2:
                self.cum_pot_evaporation = self.cum_act_evaporation
            else:
                self.cum_pot_evaporation = self.cum_act_evaporation**2 / self.boosten_soil_parameter**2

    def get_actual_evaporation(self):
        if self.net_pp < self.pot_evaporation:
            self.ea = self.net_pp + (self.cum_act_evaporation - self.cum_act_evaporation_old) / self.dt
        else:
            self.ea = self.pot_evaporation
        return self.ea
