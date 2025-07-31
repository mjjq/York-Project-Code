
import copy

from tearing_mode_solver.outer_region_solver import alfven_frequency
from tearing_mode_solver.helpers import TearingModeParameters, TimeDependentSolution

def time_scale_factor(params: TearingModeParameters,
                      si_units: bool) -> float:
    """
    Get time scale factor. By default, time is given in terms of
    alfven times. Convert to SI by enabling the relevant bool.
    """
    if si_units:
        return 1.0/alfven_frequency(
            params.R0,
            params.B0,
            params.rho0
        )
    
    return 1.0

def solution_time_scale(params: TearingModeParameters,
                        ql_sol: TimeDependentSolution,
                        si_units: bool) -> TimeDependentSolution:
    """
    Convert time units in TimeDependentSolution from alfven
    times to SI unit times
    """
    if not si_units:
        return ql_sol

    tscale_factor = time_scale_factor(params, True)

    ret: TimeDependentSolution = copy.deepcopy(ql_sol)

    ret.times = tscale_factor * ret.times
    ret.dpsi_dt = ret.dpsi_dt/tscale_factor
    ret.d2psi_dt2 = ret.d2psi_dt2/tscale_factor**2

    return ret


def time_unit_label(si_units: bool) -> str:
    """
    Get time unit label for plotting purposes
    """
    if si_units:
        return "s"
    
    return r"$\tau_A$"