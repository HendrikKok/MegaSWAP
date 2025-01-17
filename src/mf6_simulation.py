import numpy as np
from xmipy import XmiWrapper
from src.megaswap_simulation import MegaSwap
# Type hints
FloatArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray


class Logging:
    def __init__(self, ntime: int):
        self.phead = np.full((ntime, 18), np.nan)
        self.nbox = np.full(ntime, np.nan)
        self.mf6_head = np.full((ntime, 2000), np.nan)
        self.msw_head = np.full((ntime, 2000), np.nan)
        self.vsim = np.full(ntime, np.nan)
        self.qmodf = np.full((ntime, 2000), np.nan)
        self.sc1 = np.full((ntime, 2000), np.nan)
        self.sf_type = np.full((ntime, 2000), np.nan)


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
        # (f"Initialized model with {self.ncell} cells")

    def do_iter(self, sol_id: int) -> bool:
        """Execute a single iteration"""
        has_converged = self.mf6.solve(sol_id)
        return has_converged

    def update(self):
        # We cannot set the timestep (yet) in Modflow
        # -> set to the (dummy) value 0.0 for now
        self.mf6.prepare_time_step(0.0)
        self.mf6.prepare_solve(1)
        # Convergence loop
        for kiter in range(1, self.max_iter + 1):
            has_converged = self.do_iter(1)
            if has_converged:
                break
        self.mf6.finalize_solve(1)
        # Finish timestep
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
            # print(f"MF6 starting period {iperiod}")
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
            sc1, nbox, sf_type = self.msw.do_iter(self.mf6_head[0], iter)
            self.mf6_sto[0] = sc1
            has_converged = self.do_iter(1)
            if has_converged and iter > 5:
                break
            self.log_exchange_vars(iter, nbox)
        self.mf6.finalize_solve(1)
        self.msw.finalise_iter()

        # Finish timestep
        self.mf6.finalize_time_step()
        self.msw.finalise_timestep(self.mf6_head[0])  # TODO: try using self.gwl_table
        self.iperiod += 1
        current_time = self.mf6.get_current_time()
        return current_time

    def log_exchange_vars(self, iter, nbox) -> None:
        self.log.sc1[self.iperiod, iter - 1] = self.mf6_sto[0]
        self.log.msw_head[self.iperiod, iter - 1] = self.msw.gwl_table
        self.log.mf6_head[self.iperiod, iter - 1] = self.mf6_head[0]
        self.log.qmodf[self.iperiod, iter - 1] = self.msw.qmodf
        self.log.phead[self.iperiod, :] = self.msw.phead
        self.log.nbox[self.iperiod] = nbox
        self.log.vsim[self.iperiod] = self.mf6_rch[:]


def run_coupled_model(periods, mf6_parameters: dict, msw_parameters: dict):
    wdir = mf6_parameters["workdir"]
    name = mf6_parameters["model_name"]
    sim = CoupledSimulation(wdir, name, msw_parameters)
    sim.run(periods)
    sim.finalize()
    return sim.msw, sim.log

