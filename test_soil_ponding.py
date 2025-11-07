import numpy as np
import matplotlib.pyplot as plt
from src.soil import Soil
from src.ponding import Ponding

    
ntime = 100
pp = np.array([0.0]* 40 + [0.015] * 10+ [0.0]* 30 + [0.025] *20)
et = np.array([0.01]* ntime)


### test 1
soil = Soil()
aet = []
cumsum_ae = []
cumsum_ep = []
pp_log = []
et_log = []
type = []
for itime in range(ntime):
    soil.update(pp[itime], et[itime], 1.0)
    aet.append(soil.get_actual_evaporation())
    cumsum_ae.append(soil.cum_act_evaporation)
    cumsum_ep.append(soil.cum_pot_evaporation)
    pp_log.append(soil.net_pp)
    et_log.append(soil.pot_evaporation)


figure, ax = plt.subplot_mosaic(
    """
    11
    34
    """
)
ax["1"].plot(pp, label="pp") 
ax["1"].plot(et, label="et") 
ax["1"].plot(aet, '--' ,label="aet") 
ax["3"].plot(cumsum_ae, label="cumsum ae") 
ax["4"].plot(cumsum_ep, label="cumsum_ep") 
ax["1"].legend()
ax["3"].legend()
ax["4"].legend()
plt.tight_layout()
plt.savefig(r"results\bare_soil_evaporation.png")
plt.close()


### test 2
ponding = Ponding(
    zmax = 0.1, 
    area = 100.0, 
    soil_resistance=0.1,
    max_infiltration_rate=0.006, 
    dtgw=1.0, 
    factor_ponding=1.0
)

soil = Soil()
qrun = []
vol = []
qinf = []
evap = []
aet = []
for itime in range(ntime):
    ponding.add_precipitation(pp[itime])
    qinf.append(ponding.get_infiltration_flux(-3.0))
    qrun.append(ponding.get_runoff_flux())
    vol.append(ponding.volume) 
        # reset in case of ponding
    if ponding.volume > 0.0:
        soil.reset()
    else:
        soil.update(pp[itime], et[itime], 1.0)
    aet.append(soil.get_actual_evaporation())     
    evap.append(ponding.get_ponding_evaporation(et[itime])) 


figure, ax = plt.subplot_mosaic(
    """
    12
    34
    """
)
ax["1"].plot(pp, label="pp") 
ax["1"].plot(et, label="et") 
ax["2"].plot(qinf, label="qinf") 
ax["3"].plot(qrun, label="qrun") 
ax["3"].plot(vol, label="volume") 
ax["4"].plot(evap, label="ponding evaporation") 
ax["4"].plot(aet, label="soil evaporation") 
ax["1"].legend()
ax["2"].legend()
ax["3"].legend()
ax["4"].legend()

plt.tight_layout()
plt.savefig(r"results\ponding.png")
plt.close()




### test 3
ntime = 100
pp = np.array([0.025]* 40 +[0.0]* 60)
et = np.array([0.01]* ntime)


ponding = Ponding(
    zmax = 0.1, 
    area = 100.0, 
    soil_resistance=0.1,
    max_infiltration_rate=0.006, 
    dtgw=1.0, 
    factor_ponding=1.0
)
soil = Soil()
qrun = []
vol = []
qinf = []
evap = []
aet = []
for itime in range(ntime):
    ponding.add_precipitation(pp[itime])
    qinf.append(ponding.get_infiltration_flux(-3.0))
    qrun.append(ponding.get_runoff_flux())
    vol.append(ponding.volume)
    evap.append(ponding.get_ponding_evaporation(et[itime]))  

    # reset in case of ponding
    if ponding.volume > 0.0:
        soil.reset()
    else:
        soil.update(pp[itime], et[itime], 1.0)
    aet.append(soil.get_actual_evaporation())     

figure, ax = plt.subplot_mosaic(
    """
    12
    34
    """
)
ax["1"].plot(pp, label="pp") 
ax["1"].plot(et, label="et") 
ax["2"].plot(qinf, label="qinf") 
ax["3"].plot(qrun, label="qrun") 
ax["3"].plot(vol, label="volume") 
ax["4"].plot(evap, label="ponding evaporation") 
ax["4"].plot(aet, label="soil evaporation") 
ax["1"].legend()
ax["2"].legend()
ax["3"].legend()
ax["4"].legend()

plt.tight_layout()
plt.savefig(r"results\ponding_soil.png")
plt.close()
