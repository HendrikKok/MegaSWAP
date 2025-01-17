import numpy as np
import matplotlib.pyplot as plt


def plot_results(log, megaswap) -> None:
    path = "../MegaSWAP/results/"
    phead_log = log.phead
    nbox_log = log.nbox
    gwl_log = log.mf6_head
    ntime = nbox_log.size

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
    ax["4"].plot(log.qmodf[:, 4], label="qmodf")
    ax["1"].legend()
    ax["4"].legend()

    for ii in range(5):
        ax["2"].plot(log.sc1[:, ii], label=f"sc1 iter={ii}")
        # ax['4'].plot(log.sf_type[:,ii], label = 's-formulation')
    ax["2"].legend()

    ax["3"].plot(log.mf6_head[:, 4], label="mf6-heads")
    ax["3"].plot(log.msw_head[:, 4], linestyle="--", label="msw-heads")
    ax["3"].legend()

    plt.tight_layout()
    plt.savefig(path + "exchange_vars_coupled.png")
    plt.close()

    for itime in range(0, ntime, n):
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
            ax["0"].plot(itime, phead_log[itime, ibox], "ro")
        ax["0"].legend()

        ax["1"].plot(log.qmodf[itime, :], label="qmodf")
        ax["1"].legend()

        ax["2"].plot(log.sc1[itime, :], label="sc1")
        ax["4"].plot(log.sf_type[itime, :], label="s-formulation")
        ax["2"].legend()

        ax["3"].plot(log.mf6_head[itime, :], label="mf6-heads")
        ax["3"].plot(log.msw_head[itime, :], label="msw-heads")
        ax["3"].legend()
        # ax['4'].legend()

        plt.tight_layout()
        plt.savefig(path + f"exchange_vars_coupled_t{itime}.png")
        plt.close()
