from chease_tools.dr_term_at_q import CheaseColumns, read_columns
from tearing_mode_solver.helpers import TearingModeParameters
from tearing_mode_solver.outer_region_solver import (
    eta_to_lundquist_number,
    rational_surface
)
from tearing_mode_solver.profiles import value_at_r
from debug.log import logger

def get_parameters(chease_cols: CheaseColumns,
                   poloidal_mode_number: int,
                   toroidal_mode_number: int) -> TearingModeParameters:
    r_vals = chease_cols.eps / chease_cols.eps[-1]
    q_profile = list(zip(r_vals, chease_cols.q))
    j_profile = list(zip(r_vals, chease_cols.j_phi))

    B_tor = 1.0
    R_0 = 1.0
    lundquist_number = 1e7

    params = TearingModeParameters(
        poloidal_mode_number=poloidal_mode_number,
        toroidal_mode_number=toroidal_mode_number,
        lundquist_number=lundquist_number,
        initial_flux=1e-12,
        B0=B_tor,
        R0=R_0,
        q_profile=q_profile,
        j_profile=j_profile,
        r_minor=1.0
    )
    logger.debug(params)

    return params