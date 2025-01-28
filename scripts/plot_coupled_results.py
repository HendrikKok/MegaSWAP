import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imod

def plot_results(log, megaswap, ntime_) -> None:
    path = "../MegaSWAP/results/"
    phead_log = log.phead
    nbox_log = log.nbox
    gwl_log = log.mf6_head
    ntime = nbox_log.size
    
    grb_path = '../MegaSWAP' + r"\mf6_model\model.dis.grb" 
    hds_path = '../MegaSWAP' + r"\mf6_model\flow.hds" 
    cbc_path = '../MegaSWAP' + r"\mf6_model\flow.cbc" 
    heads = imod.mf6.open_hds(hds_path, grb_path)

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

    n = int(ntime / 10)
    if ntime < 10:
        n = 1
    colors = []
    for ibox in range(max_box):
        ax["0"].plot(phead_log[:, ibox], label=f"h{ibox}")
    ax["0"].legend()

    for itime in range(0, ntime, n):
        p = np.repeat(phead_log[itime, 0:max_box], 2)
        y = np.stack([box_top[0:max_box], box_bottom[0:max_box]], axis=1).ravel()
        plot = ax["1"].plot(p, y, label=f"t={itime}")
        colors.append(plot[0].get_color())
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
    for itime in range(0, ntime, n):
        head = gwl_log[itime]
        ax["3"].hlines(head, 0, 1, color=colors[icol], label=f"t={itime}")
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

    n = int(ntime / 10)
    if ntime < 10:
        n = 1
    colors = []
    for ibox in range(max_box):
        ax["0"].plot(phead_log[:, ibox], label=f"h{ibox}")
    ax["0"].legend()

    ax["1"].plot(log.vsim, label="vsim")
    ax["1"].plot(megaswap.qrch[0:ntime_], label="neerslag")
    ax["4"].plot(log.qmodf[:, 1], label="qmodf")
    ax["1"].legend()
    ax["4"].legend()

    for ii in range(5):
        ax["2"].plot(log.sc1[:, ii], label=f"sc1 iter={ii}")
        # ax['4'].plot(log.sf_type[:,ii], label = 's-formulation')
    ax["2"].legend()

    ax["3"].plot(log.mf6_head[:, 1], label="mf6-heads")
    ax["3"].plot(log.msw_head[:, 1], linestyle="--", label="msw-heads")
    ax["3"].plot(heads.isel(layer = 0, x = 0, y = 0).to_numpy()[0:ntime_], linestyle="--", label="mf6-heads_out")
    ax["3"].legend()

    plt.tight_layout()
    plt.savefig(path + "exchange_vars_coupled.png")
    plt.close()

    # for itime in range(0, ntime, n):
    #     figure, ax = plt.subplot_mosaic(
    #         """
    #         01
    #         04
    #         23
    #         """
    #     )
    #     n = int(ntime / 10)
    #     if ntime < 10:
    #         n = 1
    #     colors = []
    #     for ibox in range(max_box):
    #         ax["0"].plot(phead_log[:, ibox], label=f"h{ibox}")
    #         ax["0"].plot(itime, phead_log[itime, ibox], "ro")
    #     ax["0"].legend()
# 
    #     ax["1"].plot(log.qmodf[itime, :], label="qmodf")
    #     ax["1"].legend()
# 
    #     ax["2"].plot(log.sc1[itime, :], label="sc1")
    #     ax["4"].plot(log.sf_type[itime, :], label="s-formulation")
    #     ax["2"].legend()
# 
    #     ax["3"].plot(log.mf6_head[itime, :], label="mf6-heads")
    #     ax["3"].plot(log.msw_head[itime, :], label="msw-heads")
    #     ax["3"].legend()
    #     # ax['4'].legend()
# 
    #     plt.tight_layout()
    #     plt.savefig(path + f"exchange_vars_coupled_t{itime}.png")
    #     plt.close()


def plot_combined_results(log, megaswap, model_dir, ntime_) -> None:
    # old stuff
    svat_per = pd.read_csv(model_dir + r"\metaswap_a\msw\csv\svat_per_0000000001.csv")
    msw_sc1 = svat_per["    sc1(m3/m2/m)"][0:ntime_]
    msw_qmodf = (svat_per["       qmodf(mm)"] / 1000)[0:ntime_]  # to m
    ntime = 117
    phead_raw = np.loadtxt(model_dir + r"\metaswap_a\fort.112", skiprows=0)
    msw_phead = phead_raw[:,3].ravel().reshape((ntime,18))[0:ntime_,:]
    
    grb_path = model_dir + r"\mf6_model\model.dis.grb" 
    hds_path = model_dir + r"\mf6_model\flow.hds" 
    cbc_path = model_dir + r"\mf6_model\flow.cbc" 
    heads = imod.mf6.open_hds(hds_path, grb_path)
    cbc = imod.mf6.open_cbc(cbc_path,grb_path, flowja=True)
    msw_heads_mf6 = heads.isel(layer = 0, x = 0, y = 0).to_numpy()[0:ntime_]
    msw_vsim = (cbc['rch_rch-1'].isel(layer = 0, x = 0, y = 0).to_numpy() / 100.0)[0:ntime_]
    
    # new stuff
    path = "../MegaSWAP/results/"
    phead_log = log.phead
    nbox_log = log.nbox
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
        012
        """
    )
    for ibox in range(max_box):
        ax["0"].plot(phead_log[:, ibox][0:ntime_], label=f"h_{ibox}")
        ax["0"].plot(msw_phead[:, ibox],'--', label=f"h_msw_{ibox}")
    ax["0"].legend()
    

    ax["1"].plot(log.sc1[0:ntime_, 1] - msw_sc1[0:ntime_], label="delta sc1")
    ax["1"].legend()

    ax["2"].plot(msw_vsim[0:ntime_], label="msw_vism")
    ax["2"].plot(log.vsim[0:ntime_], label="vism")
    ax["2"].legend()
    
    
    plt.tight_layout()
    plt.savefig(path + "phead.png")
    plt.close()

    # ax["1"].plot(megaswap.qrch[0:ntime_], label="rch")
    # ax["1"].legend()
    figure, ax = plt.subplot_mosaic(
        """
        2467
        2467
        """
    )
    for iter in range(5):
        ax["4"].plot(log.qmodf[0:ntime_, iter], label=f"qmodf_msw{iter}")
    ax["4"].plot(msw_qmodf,'--', label="qmodf_msw")
    ax["4"].legend()

    for iter in range(5):
       ax["2"].plot(log.sc1[0:ntime_, iter], label=f"sc1_msw{iter}")
    ax["2"].plot(msw_sc1,'--', label="sc1_msw")
    ax["2"].legend()

    # for iter in range(5):
    #    ax["6"].plot(log.mf6_head[0:ntime_, iter], label=f"heads{iter}")
    # ax["6"].legend()
    
    for iter in range(5):
       # ax["7"].plot(log.msw_head[0:ntime_, iter], label="heads")
       ax["6"].plot(log.mf6_head[0:ntime_, iter], label=f"heads{iter}")
       ax["6"].plot(msw_heads_mf6[0:ntime], '--' ,label="heads_msw")
       
    ax["6"].legend()

    # ax["5"].plot(log.fig[0:ntime_], label="fig")
    # ax["5"].legend()
    
    ax["7"].plot(log.vsim[0:ntime_], label="vsim")
    ax["7"].plot(msw_vsim[0:ntime_],'--', label="vsim_msw")
    
    plt.tight_layout()
    plt.savefig(path + "exchange_vars_coupled_combined.png")
    plt.close()