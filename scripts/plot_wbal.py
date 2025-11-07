import numpy as np
import matplotlib.pyplot as plt

def plot_wbal(ntime, msw_log, qrch, results_path:str):
    alloc_size = qrch.shape[0]

    ##### waterbalance internals, summed
    # ds = -msw_log.ds[:,-1]
    ds = msw_log.sv_old.sum(axis=1) - msw_log.sv.sum(axis=1)
    ds_pos = np.zeros_like(ds)
    ds_pos[ds > 0.0] = ds[ds > 0.0]
    ds_neg = np.zeros_like(ds)
    ds_neg[ds < 0.0] = ds[ds < 0.0]

    qmodf = msw_log.qmodf[:,-1]
    qmodf_pos = np.zeros_like(qmodf)
    qmodf_pos[qmodf > 0.0] = qmodf[qmodf > 0.0]
    qmodf_neg = np.zeros_like(qmodf)
    qmodf_neg[qmodf < 0.0] = qmodf[qmodf < 0.0]  

    summed = qrch + ds + qmodf

    fig ,ax = plt.subplots(1)
    p = ax.bar(np.arange(ntime), ds_pos[0:ntime], label = 'ds')
    ax.bar(np.arange(ntime),qrch[0:ntime], bottom = ds_pos[0:ntime], label = 'qrch')
    p3 = ax.bar(np.arange(ntime),qmodf_pos[0:ntime], bottom = qrch[0:ntime]+ds_pos[0:ntime], label = 'qmodf')
    
    ax.bar(np.arange(ntime), ds_neg[0:ntime], color = p[0].get_facecolor())
    ax.bar(np.arange(ntime),qmodf_neg[0:ntime], bottom = ds_neg[0:ntime], color = p3[0].get_facecolor())

    
    ax.plot(np.arange(ntime),summed[0:ntime],label = 'sum', color = 'black')
    ax.legend()
    # ax.set_ylim(-0.011,0.011)
    plt.tight_layout()
    plt.savefig(results_path + "/wbal_sum.png")
    plt.close()


    fig , ax = plt.subplots(4)
    for ibox in range(4):
        if ibox ==0:
            qtop = qrch
        else:
            qtop = -msw_log.qmv[:,ibox-1]  # > 0 voor instroom

        ds = msw_log.sv_old[:,ibox] - msw_log.sv[:,ibox]
        ds_pos = np.zeros_like(ds)
        ds_pos[ds > 0.0] = ds[ds > 0.0]
        ds_neg = np.zeros_like(ds)
        ds_neg[ds < 0.0] = ds[ds < 0.0]

        q = msw_log.qmv[:,ibox]
        q_pos = np.zeros_like(q)
        q_pos[q>0] = q[q>0]
        q_neg = np.zeros_like(q)
        q_neg[q<0] = q[q<0]

        qtop_pos =  np.zeros_like(qtop)
        qtop_pos[qtop > 0] = qtop[qtop > 0]
        qtop_neg =  np.zeros_like(qtop)
        qtop_neg[qtop < 0] = qtop[qtop < 0]

        summed = ds + q + qtop

        ax[ibox].bar(np.arange(ntime), ds_pos[0:ntime], label = 'sv', color = 'green')
        ax[ibox].bar(np.arange(ntime), q_pos[0:ntime], bottom = ds_pos[0:ntime], label = 'qmv-out', color = 'orange')
        ax[ibox].bar(np.arange(ntime), qtop_pos[0:ntime], bottom = ds_pos[0:ntime] + q_pos[0:ntime], label = 'qmv-in', color = 'red')

        ax[ibox].bar(np.arange(ntime), ds_neg[0:ntime],color = 'green')
        ax[ibox].bar(np.arange(ntime), q_neg[0:ntime], bottom = ds_neg[0:ntime], color = 'orange')
        ax[ibox].bar(np.arange(ntime), qtop_neg[0:ntime], bottom = ds_neg[0:ntime] + q_neg[0:ntime], color = 'red')

        ax[ibox].plot(np.arange(ntime),summed[0:ntime],label = 'sum', color = 'black')
        ax[ibox].set_title(f'box {ibox}')
        ax[ibox].set_ylim(-0.01,0.01)
    ax[ibox].legend()
    plt.tight_layout()
    plt.savefig(results_path + "/wbal_box.png")
    plt.close()


    ##### waterbalance theta, summed
    # theta   -> m3/m3
    theta = msw_log.theta
    # theta[msw_log.active == 0] == 0.0
    ds_tot = -(np.diff(theta, axis=0)[:,] * msw_log.box_thicknes) 
    ds_tot[np.isnan(ds_tot)] = 0.0
    ds = ds_tot.sum(axis=1)

    ds_pos = np.zeros_like(ds)
    ds_pos[ds > 0.0] = ds[ds > 0.0]
    ds_neg = np.zeros_like(ds)
    ds_neg[ds < 0.0] = ds[ds < 0.0]

    vsim = np.zeros_like(ds)  # -msw_log.vsim[:-1,-1]
    vsim_pos = np.zeros_like(vsim)
    vsim_pos[vsim > 0.0] = vsim[vsim > 0.0]
    vsim_neg = np.zeros_like(vsim)
    vsim_neg[vsim < 0.0] = vsim[vsim < 0.0]

    qmodf = msw_log.qmodf[:,-1]
    qmodf_pos = np.zeros_like(qmodf)
    qmodf_pos[qmodf > 0.0] = qmodf[qmodf > 0.0]
    qmodf_neg = np.zeros_like(qmodf)
    qmodf_neg[qmodf < 0.0] = qmodf[qmodf < 0.0] 


    summed = qrch[:-1] + ds + vsim[:] + qmodf[:-1]

    fig ,ax = plt.subplots(1)
    p = ax.bar(np.arange(ntime), ds_pos[0:ntime], label = 'ds')
    ax.bar(np.arange(ntime),qrch[0:ntime], bottom = ds_pos[0:ntime], label = 'qtop')
    p2 = ax.bar(np.arange(ntime),vsim_pos[0:ntime], bottom = qrch[0:ntime]+ds_pos[0:ntime], label = 'qbot')
    p3 = ax.bar(np.arange(ntime),qmodf_pos[0:ntime], bottom = qrch[0:ntime]+ds_pos[0:ntime]+vsim_pos[0:ntime], label = 'qmodf')
    
    ax.bar(np.arange(ntime), ds_neg[0:ntime], color = p[0].get_facecolor())
    ax.bar(np.arange(ntime),vsim_neg[0:ntime], bottom = ds_neg[0:ntime], color = p2[0].get_facecolor())
    ax.bar(np.arange(ntime),qmodf_neg[0:ntime], bottom = ds_neg[0:ntime]+vsim_neg[0:ntime], color = p3[0].get_facecolor())

    ax.plot(np.arange(ntime),summed[0:ntime],label = 'sum', color = 'black')
    ax.legend()
    # ax.set_ylim(-0.011,0.011)
    plt.tight_layout()
    plt.savefig(results_path + "/wbal_theta_sum.png")
    plt.close()

    # per box
    ds_theta_sum = np.zeros(alloc_size-1)
    dikte = 0

    theta = msw_log.theta
    theta[msw_log.active == 0] == 0.0
    ds_tot = -(np.diff(theta, axis=0)[:,] * msw_log.box_thicknes)

    fig ,ax = plt.subplots(4)
    for ibox in range(4):
        if ibox ==0:
            qtop = qrch
        else:
            qtop = -msw_log.qmv[:,ibox-1]  # > 0 voor instroom
        qtop = qtop[:-1]
        ds = ds_tot[:,ibox]
        ds_theta_sum += ds 
        dikte += msw_log.box_thicknes[ibox]
        ds_pos = np.zeros_like(ds)
        ds_pos[ds > 0.0] = ds[ds > 0.0]
        ds_neg = np.zeros_like(ds)
        ds_neg[ds < 0.0] = ds[ds < 0.0]

        q = msw_log.qmv[:,ibox][:-1]
        q_pos = np.zeros_like(q)
        q_pos[q>0] = q[q>0]
        q_neg = np.zeros_like(q)
        q_neg[q<0] = q[q<0]

        qtop_pos =  np.zeros_like(qtop)
        qtop_pos[qtop > 0] = qtop[qtop > 0]
        qtop_neg =  np.zeros_like(qtop)
        qtop_neg[qtop < 0] = qtop[qtop < 0]

        summed = ds + q + qtop

        ax[ibox].bar(np.arange(ntime), ds_pos[0:ntime], label = 'dtheta', color = 'green')
        ax[ibox].bar(np.arange(ntime),q_pos[0:ntime], bottom = ds_pos[0:ntime], label = 'qmv-bot', color = 'orange')
        ax[ibox].bar(np.arange(ntime),qtop_pos[0:ntime], bottom = ds_pos[0:ntime] + q_pos[0:ntime], label = 'qmv-top', color = 'red')

        ax[ibox].bar(np.arange(ntime), ds_neg[0:ntime],color = 'green')
        ax[ibox].bar(np.arange(ntime),q_neg[0:ntime], bottom = ds_neg[0:ntime], color = 'orange')
        ax[ibox].bar(np.arange(ntime),qtop_neg[0:ntime], bottom = ds_neg[0:ntime] + q_neg[0:ntime], color = 'red')

        ax[ibox].plot(np.arange(ntime),summed[0:ntime],label = 'sum', color = 'black')
        ax[ibox].set_title(f'box {ibox}')
        ax[ibox].set_ylim(-0.0111,0.011)
    ax[ibox].legend()
    plt.tight_layout()
    plt.savefig(results_path + "/wbal_theta_box.png")
    plt.close()


