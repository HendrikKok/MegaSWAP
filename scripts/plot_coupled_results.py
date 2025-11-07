import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imod
import xarray as xr

def plot_results(log, megaswap, ntime_, max_infiltration:float, name:str = '', ntime_min = 0) -> None:
    path = f"../MegaSWAP/results/{name}/"
    phead_log = log.phead
    nbox_log = log.active.sum(axis=1)
    gwl_log = log.mf6_head
    ntime = nbox_log.size
    color_index = np.zeros(ntime, dtype=np.int32)
    
    # plot pheads
    max_box = 4
    box_top = megaswap.database.box_top
    box_bottom = megaswap.database.box_bottom
    figure, ax = plt.subplot_mosaic(
        """
        00113
        22113
        """
    )

    n = int((ntime-ntime_min) / 10)
    if ntime < 10:
        n = 1
    colors = []
    for ibox in range(max_box):
        ax["0"].plot(phead_log[:, ibox], label=f"h{ibox}")
    ax["0"].legend()

    ii=0
    for itime in range(ntime_min, ntime, n):
        p = np.repeat(phead_log[itime, 0:max_box], 2)
        y = np.stack([box_top[0:max_box], box_bottom[0:max_box]], axis=1).ravel()
        plot = ax["1"].plot(p, y, label=f"t={itime}")
        colors.append(plot[0].get_color())
        color_index[itime] =ii
        ii+1
    pmin = phead_log[np.isfinite(phead_log)].min()
    pmax = phead_log[np.isfinite(phead_log)].max()
    ax["1"].hlines(0.0, pmin, pmax, color="grey")
    for ibox in range(max_box):
        ax["1"].hlines(box_bottom[ibox], pmin, pmax, color="grey")

    ax["2"].plot(nbox_log, label="active boxes")
    ax["2"].legend()

    ax["3"].hlines(0.0, 0, 1, color="grey")
    for ibox in range(max_box):
        ax["3"].hlines(box_bottom[ibox], 0, 1, color="grey")
    icol = 0
    for itime in range(ntime_min, ntime, n):
        head = gwl_log[itime]
        ax["3"].hlines(head, 0, 1, color=colors[color_index[icol]], label=f"t={itime}")
        icol += 1
    ax["1"].legend()
    plt.tight_layout()
    plt.savefig(path + "pheads_coupled.png")
    plt.close()

    figure, ax = plt.subplot_mosaic(
        """
        01
        04
        23
        """
    )

    n = int((ntime-ntime_min) / 10)
    if ntime < 10:
        n = 1
    colors = []
    for ibox in range(max_box):
        ax["0"].plot(phead_log[ntime_min:ntime_, ibox], label=f"h{ibox}")
    ax["0"].legend()

    ax["1"].plot(log.vsim[ntime_min:ntime_,-1], label="vsim")
    ax["1"].plot(megaswap.qrch[ntime_min:ntime_], label="neerslag")
    ax["4"].plot(log.qmodf[ntime_min:ntime_,-1], label="qmodf")
    ax["1"].legend()
    ax["4"].legend()
    

    # for ii in range(5):
    ii = 0
    ax["2"].plot(log.sc1[ntime_min:ntime_,-1], label="sc1")
    ax["2"].legend()

    for iter in range(5):
        ax["3"].plot(log.msw_head[ntime_min:ntime_,iter], color = 'grey')
    iter = 0
    ax["3"].plot(log.mf6_head[ntime_min:ntime_, iter], label="mf6-heads")
    ax["3"].legend()

    plt.tight_layout()
    plt.savefig(path + "exchange_vars_coupled.png")
    plt.close()

    figure, ax = plt.subplot_mosaic(
        """
        01
        23
        """
    )
    ax["0"].plot(log.qrun[0:ntime_], label="runoff")
    ax["0"].legend()
    ax["1"].plot(log.vpond[0:ntime_], label="volume")
    ax["1"].legend()
    ax["2"].plot(log.qrch_init[0:ntime_], label="pp pot")
    ax["2"].plot(log.qrch[0:ntime_], label="inf act")
    ax["2"].plot(np.array([max_infiltration] * ntime_), label="max inf")
    ax["2"].legend()

    ax["3"].plot(log.evap_soil[0:ntime_], label="evap soil")
    ax["3"].plot(log.evap_pond[0:ntime_], label="evap pond")
    ax["3"].legend()
    plt.tight_layout()
    plt.savefig(path + "qrun.png")
    plt.close()


    





def plot_combined_results(log, megaswap, model_dir,  ntime_, ntime_min=0) -> None:
    # old stuff
    svat_per = pd.read_csv(model_dir + r"\metaswap\msw\csv\svat_per_0000000001.csv")
    msw_sc1 = svat_per["    sc1(m3/m2/m)"][ntime_min:ntime_].to_numpy()
    msw_qmodf = (svat_per["       qmodf(mm)"] / 1000)[ntime_min:ntime_].to_numpy()  # to m
    
    nrow, _ = svat_per.shape
    msw_phead = np.full((nrow,18),np.nan)
    for ibox in range(18):
        msw_phead[:,ibox] = svat_per[f'       phrz{ibox+1:02d}(m)'].to_numpy()

    grb_path = model_dir + r"\mf6_model\model.dis.grb" 
    hds_path = model_dir + r"\mf6_model\flow.hds" 
    cbc_path = model_dir + r"\mf6_model\flow.cbc" 
    heads = imod.mf6.open_hds(hds_path, grb_path)
    cbc = imod.mf6.open_cbc(cbc_path,grb_path, flowja=True)
    msw_heads_mf6 = heads.isel(layer = 0, x = 0, y = 0).to_numpy()[ntime_min:ntime_]
    msw_vsim = (cbc['rch_rch-1'].sum(dim = ['layer','y','x']).to_numpy() / 100.0)[ntime_min:ntime_]
    
    # new stuff
    path = "../MegaSWAP/results/"
    phead_log = log.phead
    nbox_log = log.active.sum(axis=1)
    gwl_log = log.mf6_head
    ntime = nbox_log.size

    # plot pheads
    max_box = 4
    # figure, ax = plt.subplot_mosaic(
    #     """
    #     01
    #     04
    #     23
    #     """
    # )
    figure, ax = plt.subplot_mosaic(
        """
        0
        0
        0
        1
        """
    )
    for ibox in range(max_box):
        ax["0"].plot(phead_log[:, ibox][ntime_min:ntime_], label=f"h_{ibox}")
        ax["0"].plot(msw_phead[:, ibox][ntime_min:ntime_],'--', label=f"h_msw_{ibox}")
    ax["0"].legend()
    
    ax["1"].plot(nbox_log[ntime_min:ntime_], label="active boxes")
    

    # ax["1"].plot(msw_sc1, label="msw_sc1")
    # ax["1"].plot(log.sc1[ntime_min:ntime_, 1], label="sc1")
    # ax["1"].legend()

    # ax["1"].plot(msw_vsim, label="msw_vism")
    # ax["1"].plot(log.vsim[ntime_min:ntime_,-1], label="vism")
    # ax["1"].legend()
    
    
    plt.tight_layout()
    plt.savefig(path + "phead_coupled_combined.png")
    plt.close()

    # ax["1"].plot(megaswap.qrch[0:ntime_], label="rch")
    # ax["1"].legend()
    figure, ax = plt.subplot_mosaic(
        """
        24
        67
        """
    )
    # for iter in range(5):
    ax["4"].plot(log.qmodf[ntime_min:ntime_, -1], label="qmodf")
    ax["4"].plot(msw_qmodf,'--', label="qmodf_msw")
    ax["4"].legend()

    # for iter in range(5):
    ax["2"].plot(log.sc1[ntime_min:ntime_, -1], label="sc1")
    ax["2"].plot(msw_sc1,'--', label="sc1_msw")
    ax["2"].legend()

    # for iter in range(5):
    #    ax["6"].plot(log.mf6_head[0:ntime_, iter], label=f"heads{iter}")
    # ax["6"].legend()
    
    #for iter in range(5):
       # ax["7"].plot(log.msw_head[0:ntime_, iter], label="heads")
    ax["6"].plot(log.mf6_head[ntime_min:ntime_, -1], label="heads")
    ax["6"].plot(msw_heads_mf6, '--' ,label="heads_msw")
       
    ax["6"].legend()

    # ax["5"].plot(log.fig[0:ntime_], label="fig")
    # ax["5"].legvsimend()
    
    ax["7"].plot(log.vsim[ntime_min:ntime_,-1], label="vsim")
    ax["7"].plot(msw_vsim,'--', label="vsim_msw")
    ax["7"].legend()
    plt.tight_layout()
    plt.savefig(path + "exchange_vars_coupled_combined.png")
    plt.close()
    
    
    figure, ax = plt.subplot_mosaic(
        """
        123
        """
    )
    niter = log.niter[ntime_min:ntime_].max()
    for iter in range(niter):
        ax["1"].plot(log.sc1[ntime_min:ntime_, iter], label="sc1")
        ax["2"].plot(log.mf6_head[ntime_min:ntime_, iter], label="heads")
    
    ax["3"].plot(log.niter[ntime_min:ntime_], label="niter")
    ax["1"].legend()
    ax["2"].legend()
    ax["3"].legend()
    plt.tight_layout()
    plt.savefig(path + "iter.png")
    plt.close()


    nbox_log = log.active.sum(axis=1)
