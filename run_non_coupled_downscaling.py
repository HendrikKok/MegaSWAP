#%%
import numpy as np
from src.mf6_simulation import run_experimental_non_coupled_model
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.colors as mcolors

nodes = xr.open_dataset(r'database\nodes_079.nc')
box = xr.open_dataset(r"database\unsa_079_100.nc") # -> 1m

ig_tb = np.arange(box.nxlig, box.nxuig + 1, 1) # box.nxlig - 1
ip_tb = np.arange(box.nlip, box.nuip + 1, 1)

phead_nodes = nodes['pheadtb'].assign_coords(
            {
                "node": nodes.node.to_numpy(), 
                "ip": ip_tb,
                "ig": ig_tb,
            }
        )
theta_nodes = nodes['thetatb'].assign_coords(
            {
                "node": nodes.node.to_numpy(), 
                "ip": ip_tb,
                "ig": ig_tb,
            }
        )

bot_nodes = np.cumsum(nodes.dz_key).to_numpy()[:-1]
z_nodes = bot_nodes + (nodes.dz_key * 0.5).to_numpy()[:-1]
bot_box = -box.hbotb.to_numpy()
box_index_nodes = np.searchsorted(bot_box,bot_nodes)


# msw inputs
qrch = np.array([0.0045] * 200) 
qpet = np.zeros_like(qrch)
qroot = np.zeros_like(qrch)

init_gwl = -3.0
gwl =  -2.89

msw_parameters = {
    "databse_path": r"database\unsa_079_100.nc",
    "rootzone_dikte": 1.0,
    "qrch": qrch,
    "qpet": qpet,
    "surface_elevation": 0.0,
    "initial_gwl": gwl,
    "initial_phead": -(0-gwl),
    "dtgw": 1.0,
    "area": 100.0,
    "max_infiltration": 0.0032, # 0.0036  
    "qtree": qroot,
    "factor_ponding": 1.0,
}

# mf6 inputs
mf6_parameters = {
    "workdir": r"c:\src\MegaSWAP\mf6_model",
    "model_name": "model",
}

# chek s_mf6
ntime = 150 # 120
d1 = 0.0
d2 = 0.02
megaswap, log = run_experimental_non_coupled_model(ntime, msw_parameters, d1)

#%%
ip = log.ip[0:ntime,:].astype(dtype=np.int32)
fip = log.fip[0:ntime,:]
ig = log.ig[0:ntime].astype(dtype=np.int32)
fig = log.fig[0:ntime]

#%%
nbox = 3
nleg = ntime / 6
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(0, ntime)
colors = [cmap(norm(i)) for i in range(ntime)]
node_nr = phead_nodes.node.to_numpy()[:-1]
phead_downscaled = np.full_like(node_nr, np.nan, dtype =np.float32)
theta_downscaled = np.full_like(node_nr, np.nan, dtype =np.float32)
fig1 ,ax = plt.subplots(nbox,2)
 # active.sum()
ileg = 0
for itime in range(ntime):
    # reset
    phead_downscaled[:] = np.nan
    theta_downscaled[:] = np.nan
    # time interpolation
    p1 = (phead_nodes.sel(ig=ig[itime], ip=ip[itime,:]) + (
        fig[itime] * (phead_nodes.sel(ig=ig[itime]+1, ip=ip[itime,:]) - 
                      phead_nodes.sel(ig=ig[itime], ip=ip[itime,:])
                      )
    )).to_numpy()
    p2 = (phead_nodes.sel(ig=ig[itime], ip=ip[itime,:]+1) + (
        fig[itime] * (phead_nodes.sel(ig=ig[itime]+1, ip=ip[itime,:]+1) - 
                      phead_nodes.sel(ig=ig[itime], ip=ip[itime,:]+1)
                      )
    )).to_numpy()
    t1 = (theta_nodes.sel(ig=ig[itime], ip=ip[itime,:]) + (
        fig[itime] * (theta_nodes.sel(ig=ig[itime]+1, ip=ip[itime,:]) - 
                      theta_nodes.sel(ig=ig[itime], ip=ip[itime,:])
                      )
    )).to_numpy()
    t2 = (theta_nodes.sel(ig=ig[itime], ip=ip[itime,:]+1) + (
        fig[itime] * (theta_nodes.sel(ig=ig[itime]+1, ip=ip[itime,:]+1) - 
                      theta_nodes.sel(ig=ig[itime], ip=ip[itime,:]+1)
                      )
    )).to_numpy()

    for ibox in range(nbox): 
        active = log.active[itime,ibox] == 1
        if active: 
            # box interpolation
            active_nodes = node_nr[box_index_nodes == ibox]
            node_slice = slice(active_nodes[0], active_nodes[-1]+1)
            phead_downscaled[active_nodes]=(p1[:,ibox][active_nodes] + fip[itime,ibox+1]*(p2[:,ibox]-p1[:,ibox])[active_nodes])
            phead_downscaled[active_nodes] = phead_downscaled[active_nodes][0]- z_nodes[active_nodes][0]
            theta_downscaled[active_nodes]=(t1[:,ibox][active_nodes] + fip[itime,ibox+1]*(t2[:,ibox]-t1[:,ibox])[active_nodes])
            ax[ibox,0].scatter(theta_downscaled[active_nodes][-1], phead_downscaled[active_nodes][-1], color=colors[itime],s=10)  
            ax[ibox,1].scatter(log.theta[itime,ibox],log.phead[itime,ibox], color = colors[itime],s=10, label = f"t={itime}")
            ax[ibox,0].set_ylim(-2.5,0)
            ax[ibox,1].set_ylim(-2.5,0)
            ax[ibox,0].set_xlim(0.1,0.35)
            ax[ibox,1].set_xlim(0.1,0.35)
            ax[ibox,0].invert_yaxis()
            ax[ibox,1].invert_yaxis()
            ax[ibox,0].set_ylabel(f'phead {ibox}')
# plt.plot(bot_nodes, phead_downscaled,label ='downscaled')
# plt.plot(bot_box[active_mask[0,:]], log10.phead[itime,:][active_mask[0,:]])
# plt.legend()
h, l = ax[nbox-1,1].get_legend_handles_labels()
h = np.array(h)
l = np.array(l)
index = np.arange(0,ntime, int(nleg), dtype = np.int32)

# ax[nbox-1,1].legend(list(h[index]),list(l[index]))

ax[nbox-1,0].set_xlabel('theta')
ax[nbox-1,1].set_xlabel('theta')
ax[0,0].set_title('node-level')
ax[0,1].set_title('box-level')

# ax[1].legend()
#ax[2].legend()
plt.tight_layout()
plt.savefig(r"c:\src\MegaSWAP\results\theta_phead_downscaled.png")
plt.close()



#%%  phead profiel met punten voor gemiddelde
gws = log.mf6_head[:,0]
ntime = 150
nbox = 3
fig1 ,ax = plt.subplots(nbox, 2)
box_index_nodes
node_nr = phead_nodes.node.to_numpy()[:-1]
phead_downscaled = np.full_like(node_nr, np.nan, dtype =np.float32)
theta_downscaled = np.full_like(node_nr, np.nan, dtype =np.float32)
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(0, ntime)
colors = [cmap(norm(i)) for i in range(ntime)]
for itime in range(ntime):
    # reset
    phead_downscaled[:] = np.nan
    active = log.active[itime,:] == 1
    p1 = (phead_nodes.sel(ig=ig[itime], ip=ip[itime,:]) + (
        fig[itime] * (phead_nodes.sel(ig=ig[itime]+1, ip=ip[itime,:]) - 
                      phead_nodes.sel(ig=ig[itime], ip=ip[itime,:])
                      )
    )).to_numpy()
    p2 = (phead_nodes.sel(ig=ig[itime], ip=ip[itime,:]+1) + (
        fig[itime] * (phead_nodes.sel(ig=ig[itime]+1, ip=ip[itime,:]+1) - 
                      phead_nodes.sel(ig=ig[itime], ip=ip[itime,:]+1)
                      )
    )).to_numpy()
    for ibox in range(nbox):
        active = log.active[itime,ibox] == 1  #  - z_nodes[active_nodes]
        if active: 
            active_nodes = node_nr[box_index_nodes == ibox]    
            phead_downscaled[active_nodes]=(p1[:,ibox][active_nodes] + fip[itime,ibox]*(p2[:,ibox]-p1[:,ibox])[active_nodes])
            # theta_downscaled[active_nodes]=(t1.sel(ip=ibox)[active_nodes] + fip[itime,ibox]*(t2.sel(ip=ibox)-t1.sel(ip=ibox))[active_nodes])
            ax[ibox,0].plot(phead_downscaled[active_nodes],-bot_nodes[active_nodes], color=colors[itime])
            ymin,ymax = ax[ibox,0].get_ylim()
            ax[ibox,0].scatter(log.phead[itime,ibox],-bot_box[ibox], color = colors[itime]) 
            # ax[ibox,0].vlines(log10.phead[itime,ibox],ymin,ymax, color = colors[itime], linestyles = 'dotted') 
            ax[ibox,0].set_ylabel(f'z box: {ibox}')
            ax[ibox,0].set_xlim(-4.0,0.0)

            # ax[ibox,1].scatter(itime, gws[itime],color = colors[itime], label = f"t {itime}")
            ax[ibox,1].plot(phead_downscaled[active_nodes]- z_nodes[active_nodes],-bot_nodes[active_nodes], color=colors[itime])
            ax[ibox,1].scatter(log.phead[itime,ibox],-bot_box[ibox], color = colors[itime]) 
            
            ax[ibox,1].set_ylim(ymin,ymax)
            ax[ibox,1].set_xlim(-4.0,0.0)

h, l = ax[0,1].get_legend_handles_labels()
h = np.array(h)
l = np.array(l)
index = np.arange(0,ntime, int(nleg), dtype = np.int32)
# ax[0,1].legend(list(h[index]),list(l[index]))

ax[0,0].set_title("phead")
ax[0,1].set_title("phead - z")
ax[nbox-1,0].set_xlabel('phead')
ax[nbox-1,1].set_xlabel('time')
plt.tight_layout()
plt.savefig(r"c:\src\MegaSWAP\results\phead_profile_a.png")
plt.close()

#%%  phead profiel met vline voor gemiddelde
gws = log.mf6_head[:,0]
ntime = 150 #150
nbox = 3
fig1 ,ax = plt.subplots(nbox, 2)
box_index_nodes
node_nr = phead_nodes.node.to_numpy()[:-1]
phead_downscaled = np.full_like(node_nr, np.nan, dtype =np.float32)
theta_downscaled = np.full_like(node_nr, np.nan, dtype =np.float32)
for itime in range(ntime):
    # reset
    phead_downscaled[:] = np.nan
    active = log.active[itime,:] == 1
    p1 = (phead_nodes.sel(ig=ig[itime], ip=ip[itime,:]) + (
        fig[itime] * (phead_nodes.sel(ig=ig[itime]+1, ip=ip[itime,:]) - 
                      phead_nodes.sel(ig=ig[itime], ip=ip[itime,:])
                      )
    )).to_numpy()
    p2 = (phead_nodes.sel(ig=ig[itime], ip=ip[itime,:]+1) + (
        fig[itime] * (phead_nodes.sel(ig=ig[itime]+1, ip=ip[itime,:]+1) - 
                      phead_nodes.sel(ig=ig[itime], ip=ip[itime,:]+1)
                      )
    )).to_numpy()
    for ibox in range(nbox):
        active = log.active[itime,ibox] == 1
        if active: 
            active_nodes = node_nr[box_index_nodes == ibox]    
            phead_downscaled[active_nodes]=(p1[:,ibox][active_nodes] + fip[itime,ibox]*(p2[:,ibox]-p1[:,ibox])[active_nodes])
            # theta_downscaled[active_nodes]=(t1.sel(ip=ibox)[active_nodes] + fip[itime,ibox]*(t2.sel(ip=ibox)-t1.sel(ip=ibox))[active_nodes])
            ax[ibox,0].plot(phead_downscaled[active_nodes] - z_nodes[active_nodes],-bot_nodes[active_nodes], color=colors[itime])
            # ax[ibox,0].scatter(log10.phead[itime,ibox],-bot_box[ibox], color = colors[itime]) 
            ax[ibox,1].vlines(log.phead[itime,ibox],ymin,ymax, color = colors[itime], label = f"t {itime}") 
            ax[ibox,0].set_ylabel(f'z box: {ibox}')
            ax[ibox,0].set_xlim(-4.0,0.0)
            ax[ibox,1].set_xlim(-4.0,0.0)
            # ax[ibox,1].scatter(itime, gws[itime],color = colors[itime], label = f"t {itime}")
            ax[ibox,1].set_ylim(ymin,ymax)

h, l = ax[0,1].get_legend_handles_labels()
h = np.array(h)
l = np.array(l)
index = np.arange(0,ntime, int(nleg), dtype = np.int32)
ax[0,1].legend(list(h[index]),list(l[index]))

ax[0,0].set_title('node-level')
ax[0,1].set_title('box-level')
ax[nbox-1,0].set_xlabel('phead')
ax[nbox-1,1].set_xlabel('phead')
plt.tight_layout()
plt.savefig(r"c:\src\MegaSWAP\results\phead_profile_b.png")
plt.close()

#%%
ntime = 150
nbox = 3
fig1 ,ax = plt.subplots(nbox,2)
box_index_nodes
node_nr = phead_nodes.node.to_numpy()[:-1]
phead_downscaled = np.full_like(node_nr, np.nan, dtype =np.float32)
theta_downscaled = np.full_like(node_nr, np.nan, dtype =np.float32)
tmin = 1.0
tmax = 0.0
for itime in range(ntime):
    active = log.active[itime,:] == 1
    t1 = (theta_nodes.sel(ig=ig[itime], ip=ip[itime,:]) + (
        fig[itime] * (theta_nodes.sel(ig=ig[itime]+1, ip=ip[itime,:]) - 
                      theta_nodes.sel(ig=ig[itime], ip=ip[itime,:])
                      )
    )).to_numpy()
    t2 = (theta_nodes.sel(ig=ig[itime], ip=ip[itime,:]+1) + (
        fig[itime] * (theta_nodes.sel(ig=ig[itime]+1, ip=ip[itime,:]+1) - 
                      theta_nodes.sel(ig=ig[itime], ip=ip[itime,:]+1)
                      )
    )).to_numpy()
    for ibox in range(nbox):  # active.sum()
        active_nodes = node_nr[box_index_nodes == ibox]  
        # phead_downscaled[active_nodes]=(p1.sel(ip=ibox)[active_nodes] + fip[itime,ibox]*(p2.sel(ip=ibox)-p1.sel(ip=ibox))[active_nodes])
        theta_downscaled =(t1[:,ibox][active_nodes] + fip[itime,ibox+1]*(t2[:,ibox]-t1[:,ibox])[active_nodes])
        ax[ibox,0].plot(theta_downscaled,-bot_nodes[active_nodes], label = f"t {itime}",color = colors[itime])
        ymin,ymax = ax[ibox,0].get_ylim()
        ax[ibox,1].vlines(log.theta[itime,ibox],ymin,ymax, color = colors[itime]) 
        # ax[ibox,1].scatter(log.theta[itime,ibox],-bot_box[ibox], color = colors[itime]) 
        ax[ibox,0].set_ylabel('z')
        tmin = min(tmin, theta_downscaled.min())
        tmax = max(tmax, theta_downscaled.max())
ax[nbox-1,0].set_xlabel('theta')
ax[nbox-1,1].set_xlabel('theta')
for ibox in range(nbox):
    ax[ibox,0].set_xlim(tmin, tmax)
    ax[ibox,1].set_xlim(tmin, tmax)
ax[0,0].set_title('node-level')
ax[0,1].set_title('box-level')

ax[nbox-1,1].legend()
#ax[2].legend()
plt.tight_layout()
plt.savefig(r"c:\src\MegaSWAP\results\theta_profile.png")
plt.close()


#%%