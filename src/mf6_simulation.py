import numpy as np
from xmipy import XmiWrapper
from src.msw_simulation import MegaSwap, MegaSwapExperimental

class Logging:
    def __init__(self, ntime: int):
        self.phead = np.full((ntime, 18), np.nan)
        self.nbox = np.full(ntime, np.nan)
        self.mf6_head = np.full((ntime, 2000), np.nan)
        self.msw_head = np.full((ntime, 2000), np.nan)
        self.vsim = np.full((ntime,2000), np.nan)
        self.qmodf = np.full((ntime,2000), np.nan)
        self.sc1 = np.full((ntime, 2000), np.nan)
        self.sf_type = np.full((ntime, 2000), np.nan)
        self.ig = np.full(ntime, np.nan)
        self.fig = np.full(ntime, np.nan)
        self.s = np.full(ntime, np.nan)
        self.s_old = np.full(ntime, np.nan)
        self.ds = np.full((ntime, 2000), np.nan)
        self.vcor = np.full(ntime, np.nan)
        self.niter = np.full(ntime, 0)
        self.qmv = np.full((ntime, 18), np.nan)
        self.qrun = np.full(ntime, 0.0)
        self.vpond = np.full(ntime, 0.0)
        self.qmax = np.full(ntime, 0.0)
        self.qrch_init = np.full(ntime, 0.0)
        self.qrch = np.full(ntime, 0.0)
        self.evap_soil = np.full(ntime, 0.0)
        self.evap_pond = np.full(ntime, 0.0)
        self.ig = np.full(ntime, np.nan)
        self.fig = np.full(ntime, np.nan)
        self.ip = np.full((ntime, 18), np.nan)
        self.fip = np.full((ntime,18), np.nan)
        self.sv = np.full((ntime, 18), np.nan)
        self.sv_old = np.full((ntime, 18), np.nan)
        self.theta = np.full((ntime, 18), np.nan)
        self.svtb = np.full((ntime, 18), np.nan)
        self.active = np.full((ntime, 18), np.nan)
        self.box_thicknes = np.full((18), np.nan)
        self.sigma = np.full((ntime, 18), np.nan)

class Simulation:
    """
    Run all stress periods in a simulation
    """

    def __init__(self, wdir: str, name: str):
        self.modelname = name
        self.mf6 = XmiWrapper(lib_path="libmf6.dll", working_directory=wdir)
        self.mf6.initialize()
        self.max_iter = self.mf6.get_value_ptr("SLN_1/MXITER")[0]
        shape = np.zeros(1, dtype=np.int32)
        self.ncell = self.mf6.get_grid_shape(1, shape)[0]

    def do_iter(self, sol_id: int) -> bool:
        """Execute a single iteration"""
        has_converged = self.mf6.solve(sol_id)
        return has_converged

    def update(self):
        self.mf6.prepare_time_step(0.0)
        self.mf6.prepare_solve(1)
        # Convergence loop
        for kiter in range(1, self.max_iter + 1):
            has_converged = self.do_iter(1)
            if has_converged:
                break
        self.mf6.finalize_solve(1)
        self.mf6.finalize_time_step()
        current_time = self.mf6.get_current_time()
        return current_time

    def get_times(self):
        """Return times"""
        return (
            self.mf6.get_start_time(),
            self.mf6.get_current_time(),
            self.mf6.get_end_time(),
        )

    def run(self, periods):
        iperiod = 0
        _, current_time, end_time = self.get_times()
        while (current_time < end_time) and iperiod < periods:
            current_time = self.update()
            iperiod += 1
        print(f"Simulation terminated normally for {periods} periods")

    def finalize(self):
        self.mf6.finalize()


class CoupledSimulation(Simulation):
    """
    Run all stress periods in a simulation
    """

    def __init__(self, wdir: str, name: str, msw_parameters: dict):
        super().__init__(wdir, name)
        self.mf6_head = self.mf6.get_value_ptr(f"{name.upper()}/X")
        self.mf6_sto = self.mf6.get_value_ptr(f"{name.upper()}/STO/SS")
        self.mf6_rch = self.mf6.get_value_ptr(f"{name.upper()}/RCH-1/RECHARGE")

        self.msw = MegaSwap(msw_parameters)
        self.log = Logging(msw_parameters["qrch"].size)
        self.iperiod = 0

    def update(self):
        self.mf6.prepare_time_step(0.0)
        vsim = self.msw.prepare_timestep(self.iperiod)
        self.mf6_rch[:] = vsim
        
        self.mf6.prepare_solve(1)
        # Convergence loop
        for iter in range(1, self.max_iter + 1):
            sc1 = self.msw.do_iter(iter)
            self.mf6_sto[0] = sc1
            has_converged = self.do_iter(1)
            self.msw.finalise_iter(self.mf6_head[0])
            nbox = self.msw.unsaturated_zone.non_submerged_boxes.size
            self.log_exchange_vars(iter -1, nbox, iter)
            if has_converged:
                break
        self.mf6.finalize_solve(1)

        self.mf6.finalize_time_step()
        self.msw.finalise_timestep()
        self.log_exchange_vars(-1, nbox, iter)
        self.iperiod += 1
        current_time = self.mf6.get_current_time()
        return current_time

    def log_exchange_vars(self, iter, nbox, niter_) -> None:
        self.log.sc1[self.iperiod, iter] = self.mf6_sto[0]
        self.log.msw_head[self.iperiod, iter] = self.msw.storage_formulation.gwl_table
        self.log.mf6_head[self.iperiod, iter] = self.mf6_head[0]
        self.log.qmodf[self.iperiod, iter] = self.msw.storage_formulation.qmodf
        self.log.phead[self.iperiod, :] = self.msw.unsaturated_zone.phead
        self.log.nbox[self.iperiod] = nbox
        self.log.vsim[self.iperiod] = self.msw.vsim
        self.log.s[self.iperiod] = self.msw.storage_formulation.s
        self.log.s_old[self.iperiod] = self.msw.storage_formulation.s_old
        self.log.niter[self.iperiod] = niter_
        self.log.ds[self.iperiod, iter] = self.msw.ds
        self.log.qmv[self.iperiod, 0:4] = self.msw.unsaturated_zone.qmv[0:4]
        self.log.active[self.iperiod,:] = self.msw.unsaturated_zone.active[:]
        self.log.ig[self.iperiod] = self.msw.unsaturated_zone.ig_table
        self.log.fig[self.iperiod] = self.msw.unsaturated_zone.fig_table
        self.log.ip[self.iperiod,:] = self.msw.unsaturated_zone.ip[:]
        self.log.fip[self.iperiod,:] = self.msw.unsaturated_zone.fip[:]

class NonCoupledSimulation:
    """
    Run all stress periods in a simulation
    """

    def __init__(self, msw_parameters: dict, delta: float):
        self.mf6_head = np.ones(5) * msw_parameters["initial_gwl"]
        self.mf6_sto = np.ones(5) * 0.001
        self.mf6_rch = np.zeros(1)
        self.msw = MegaSwap(msw_parameters)
        self.log = Logging(msw_parameters["qrch"].size)
        self.iperiod = 0
        self.delta = delta

    def run(self, periods):
        iperiod = 0
        while iperiod < periods:
            iperiod = self.update()
        print(f"Simulation terminated normally for {periods} periods")
        
    def update(self):
        vsim = self.msw.prepare_timestep(self.iperiod)
        self.mf6_rch[:] = vsim

        # Convergence loop
        for iter in range(1, 7):
            sc1 = self.msw.do_iter(iter)
            self.mf6_sto[0] = sc1
            has_converged = False
            if iter > 6:
                has_converged = True
            # fictive update of heads
            if iter == 1:
                self.mf6_head[0] += self.delta
                self.mf6_head[0] = min(self.mf6_head[0], 0.0)
            self.msw.finalise_iter(self.mf6_head[0])
            nbox = self.msw.unsaturated_zone.non_submerged_boxes.size
            self.log_exchange_vars(iter -1, nbox, iter)
            if has_converged:
                break
        self.msw.finalise_timestep()
        self.log_exchange_vars(iter, nbox, 999)
        self.iperiod += 1
        return self.iperiod

    def log_exchange_vars(self, iter, nbox, niter_) -> None:
        self.log.sc1[self.iperiod, iter] = self.mf6_sto[0]
        self.log.msw_head[self.iperiod, iter] = self.msw.storage_formulation.gwl_table
        self.log.mf6_head[self.iperiod, iter] = self.mf6_head[0]
        self.log.qmodf[self.iperiod, iter] = self.msw.storage_formulation.qmodf
        self.log.phead[self.iperiod, :] = self.msw.unsaturated_zone.phead
        self.log.nbox[self.iperiod] = nbox
        self.log.vsim[self.iperiod] = self.mf6_rch[:]
        self.log.s[self.iperiod] = self.msw.storage_formulation.s
        self.log.s_old[self.iperiod] = self.msw.storage_formulation.s_old
        self.log.niter[self.iperiod] = niter_
        self.log.ds[self.iperiod, iter] = self.msw.ds
        self.log.qmv[self.iperiod, iter, 0:4] = self.msw.unsaturated_zone.qmv[0:4]
        self.log.qrun[self.iperiod] = self.msw.qrun

class CoupledExperimentalSimulation(Simulation):
    """
    Run all stress periods in a simulation
    """

    def __init__(self, wdir: str, name: str, msw_parameters: dict):
        super().__init__(wdir, name)
        self.mf6_head = self.mf6.get_value_ptr(f"{name.upper()}/X")
        self.mf6_sto = self.mf6.get_value_ptr(f"{name.upper()}/STO/SS")
        self.mf6_rch = self.mf6.get_value_ptr(f"{name.upper()}/RCH-1/RECHARGE")
        self.mf6_sof_elev = self.mf6.get_value_ptr(f"{name.upper()}/SOF/ELEV")
        self.mf6_sof_cond = self.mf6.get_value_ptr(f"{name.upper()}/SOF/COND ")
        self.msw = MegaSwapExperimental(msw_parameters)
        self.log = Logging(msw_parameters["qrch"].size)
        self.iperiod = 0
        sof_cond, sof_elev = self.msw.get_sof_parameters()
        self.mf6_sof_cond[0] = sof_cond
        self.mf6_sof_elev[0] = sof_elev
        self.log.qrch_init = np.copy(msw_parameters["qrch"])
        self.log.qrch = np.zeros_like(self.log.qrch_init)

    def update(self):
        self.mf6.prepare_time_step(0.0)
        qmv_old = np.copy(self.msw.unsaturated_zone.qmv)
        vsim, sc1 = self.msw.prepare_timestep(self.iperiod, self.mf6_head[0], 0.0)
        self.msw.unsaturated_zone.qmv = np.copy(qmv_old)
        self.mf6_rch[:] = vsim
        self.mf6_sto[0] = sc1
        self.mf6.prepare_solve(1)
        mf6_head_old = np.copy(self.mf6_head[0])
        qmodf = 0.0
        # Convergence loop
        for iter in range(1, self.max_iter + 1):
            if iter <20:
                vsim, sc1 = self.msw.do_iter(self.iperiod, self.mf6_head[0])
                self.msw.unsaturated_zone.qmv = np.copy(qmv_old) # reset qmv inside loop
                self.mf6_rch[:] = vsim
                self.mf6_sto[0] = sc1
            has_converged = self.do_iter(1)
            # self.msw.finalise_timestep(self.mf6_head[0], qmodf, False)
            nbox = self.msw.unsaturated_zone.non_submerged_boxes.size
            self.log_exchange_vars(iter -1, nbox, iter, qmodf, vsim)
            if has_converged:
                break
        qmodf = ((self.mf6_head[0] - mf6_head_old) * sc1) - (vsim)
        self.mf6.finalize_solve(1)
        self.mf6.finalize_time_step()
        self.msw.finalise_timestep(self.mf6_head[0], qmodf, True)
        self.msw.get_thetas() # ig index updated in finalise_timestep only
        self.log_exchange_vars(-1, nbox, iter, qmodf, vsim)
        self.iperiod += 1
        current_time = self.mf6.get_current_time()
        return current_time

    def log_exchange_vars(self, iter, nbox, niter, qmodf, vsim) -> None:
        self.log.sc1[self.iperiod, iter] = self.mf6_sto[0]
        self.log.msw_head[self.iperiod, iter] = self.msw.storage_formulation.gwl_table
        self.log.mf6_head[self.iperiod, iter] = self.mf6_head[0]
        self.log.qmodf[self.iperiod, iter] = qmodf
        self.log.phead[self.iperiod, :] = self.msw.unsaturated_zone.phead
        self.log.nbox[self.iperiod] = nbox
        self.log.vsim[self.iperiod] = vsim
        self.log.s[self.iperiod] = self.msw.storage_formulation.s
        self.log.s_old[self.iperiod] = self.msw.storage_formulation.s_old
        self.log.niter[self.iperiod] = niter
        self.log.ds[self.iperiod, iter] = self.msw.ds
        # self.log.qmv[self.iperiod, iter, 0:4] = self.msw.unsaturated_zone.qmv[0:4]
        self.log.qrun[self.iperiod] = self.msw.qrun
        self.log.vpond[self.iperiod] = self.msw.ponding.volume
        self.log.qmax[self.iperiod] = self.msw.qmax
        self.log.qrch[self.iperiod] = self.msw.qrch[self.iperiod]
        self.log.evap_soil[self.iperiod] = self.msw.evap_soil
        self.log.evap_pond[self.iperiod] = self.msw.evap_ponding
        self.log.active[self.iperiod,:] = self.msw.unsaturated_zone.active[:]
        self.log.ig[self.iperiod] = self.msw.unsaturated_zone.ig_table
        self.log.fig[self.iperiod] = self.msw.unsaturated_zone.fig_table
        self.log.ip[self.iperiod,:] = self.msw.unsaturated_zone.ip[:]
        self.log.fip[self.iperiod,:] = self.msw.unsaturated_zone.fip[:]
        self.log.sigma[self.iperiod,:] = self.msw.unsaturated_zone.sigma[:]
        self.log.sv[self.iperiod,:] = self.msw.unsaturated_zone.sv[:]

class NonCoupledExperimentalSimulation:
    """
    Run all stress periods in a simulation
    """

    def __init__(self, msw_parameters: dict, delta: float):
        self.mf6_head = np.ones(5) * msw_parameters["initial_gwl"]
        self.mf6_sto = np.ones(5) * 0.001
        self.mf6_rch = np.zeros(1)
        self.mf6_sof_elev = np.zeros(1)
        self.mf6_sof_cond = np.ones(1)
        self.msw = MegaSwapExperimental(msw_parameters)
        self.log = Logging(msw_parameters["qrch"].size)
        self.iperiod = 0
        sof_cond, sof_elev = self.msw.get_sof_parameters()
        self.mf6_sof_cond[0] = sof_cond
        self.mf6_sof_elev[0] = sof_elev
        self.log.qrch_init = np.copy(msw_parameters["qrch"])
        self.log.qrch = np.zeros_like(self.log.qrch_init)
        self.delta = delta
        self.qmodf = 0.0
        if "gwl" in msw_parameters.keys():
            self.gwl = msw_parameters["gwl"]
        else:
            self.gwl = None
        self.qtree = 0.0


    def update(self):
        qmv_old = np.copy(self.msw.unsaturated_zone.qmv)
        vsim, sc1 = self.msw.prepare_timestep(self.iperiod, self.mf6_head[0], self.qtree)
        self.msw.unsaturated_zone.qmv = np.copy(qmv_old)
        self.mf6_rch[:] = vsim
        self.mf6_sto[0] = sc1
        mf6_head_old = np.copy(self.mf6_head[0])
        self.qmodf = 0.0
        # Convergence loop
        for iter in range(1):
            vsim, sc1 = self.msw.do_iter(self.iperiod, self.mf6_head[0])
            self.msw.unsaturated_zone.qmv = np.copy(qmv_old) # reset qmv inside loop
            self.mf6_rch[:] = vsim
            self.mf6_sto[0] = sc1
            if self.gwl is None:
                self.mf6_head[0] += self.delta
                self.mf6_head[0] = min(self.mf6_head[0], 0.0)
            has_converged = True
            nbox = self.msw.unsaturated_zone.non_submerged_boxes.size
            self.log_exchange_vars(iter, nbox, iter, self.qmodf, vsim,sc1)
            if has_converged:
                break
        if self.gwl is None:
            self.mf6_head[0] += self.delta
            self.mf6_head[0] = min(self.mf6_head[0], 0.0)
        else:
            self.mf6_head[0] = self.gwl[self.iperiod]
        self.qmodf = ((self.mf6_head[0] - mf6_head_old) * sc1) - (vsim)
        self.log.sv_old[self.iperiod,:] = self.msw.unsaturated_zone.sv_old[:]  # before resetting sv_old in finalise_timestep
        self.msw.finalise_timestep(self.mf6_head[0], self.qmodf, True)
        self.msw.get_thetas()
        self.log_exchange_vars(-1, nbox, -1, self.qmodf, vsim,sc1)# final state at last position in arrays

        self.log.box_thicknes = self.msw.get_box_thickness()
        self.iperiod += 1
        return self.iperiod
    
    def run(self, periods):
        iperiod = 0
        while iperiod < periods:
            print(f"Running period {iperiod + 1} of {periods}")
            iperiod = self.update()
        print(f"Simulation terminated normally for {periods} periods")

    def log_exchange_vars(self, iter, nbox, niter, qmodf, vsim, sc1) -> None:
        self.log.sc1[self.iperiod, iter] = self.mf6_sto[0]
        self.log.msw_head[self.iperiod, iter] = self.msw.storage_formulation.gwl_table
        self.log.mf6_head[self.iperiod, iter] = self.mf6_head[0]
        self.log.qmodf[self.iperiod,iter] = qmodf
        self.log.phead[self.iperiod, :] = self.msw.unsaturated_zone.phead
        self.log.nbox[self.iperiod] = nbox
        self.log.vsim[self.iperiod,iter] = vsim
        self.log.s[self.iperiod] = self.msw.storage_formulation.s
        self.log.s_old[self.iperiod] = self.msw.storage_formulation.s_old
        self.log.niter[self.iperiod] = niter
        self.log.ds[self.iperiod, iter] = self.msw.ds
        self.log.qmv[self.iperiod, :] = self.msw.unsaturated_zone.qmv[:]
        self.log.qrun[self.iperiod] = self.msw.qrun
        self.log.vpond[self.iperiod] = self.msw.ponding.volume
        self.log.qmax[self.iperiod] = self.msw.qmax
        self.log.qrch[self.iperiod] = self.msw.qrch[self.iperiod]
        self.log.evap_soil[self.iperiod] = self.msw.evap_soil
        self.log.evap_pond[self.iperiod] = self.msw.evap_ponding
        self.log.ig[self.iperiod] = self.msw.unsaturated_zone.ig_table
        self.log.fig[self.iperiod] = self.msw.unsaturated_zone.fig_table
        self.log.ip[self.iperiod,:] = self.msw.unsaturated_zone.ip
        self.log.fip[self.iperiod,:] = self.msw.unsaturated_zone.fip
        self.log.sv[self.iperiod,:] = self.msw.unsaturated_zone.sv[:]
        
        self.log.theta[self.iperiod,:] = self.msw.theta[:]
        self.log.svtb[self.iperiod,:] = self.msw.svtb_log[:]
        self.log.active[self.iperiod,:] = self.msw.unsaturated_zone.active[:]
        self.log.sc1[self.iperiod,iter] = sc1


def run_coupled_model(periods, mf6_parameters: dict, msw_parameters: dict):
    wdir = mf6_parameters["workdir"]
    name = mf6_parameters["model_name"]
    sim = CoupledSimulation(wdir, name, msw_parameters)
    sim.run(periods)
    sim.finalize()
    return sim.msw, sim.log

def run_experimental_coupled_model(periods, mf6_parameters: dict, msw_parameters: dict):
    wdir = mf6_parameters["workdir"]
    name = mf6_parameters["model_name"]
    sim = CoupledExperimentalSimulation(wdir, name, msw_parameters)
    sim.run(periods)
    sim.finalize()
    return sim.msw, sim.log

def run_experimental_non_coupled_model(periods, msw_parameters: dict, delta: float):
    sim = NonCoupledExperimentalSimulation(msw_parameters, delta)
    sim.run(periods)
    return sim.msw, sim.log

def run_non_coupled_model(periods, msw_parameters: dict, delta: float):
    sim = NonCoupledSimulation(msw_parameters, delta)
    sim.run(periods)
    return sim.msw, sim.log