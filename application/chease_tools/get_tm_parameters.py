import numpy as np

from tearing_mode_solver.bootstrap import ntm_bootstrap_term
from tearing_mode_solver.ggj import ntm_ggj_term
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

def ggj_term(w: float,
             poloidal_mode_number: float,
             toroidal_mode_number: float,
             chease_cols: CheaseColumns,
             w_d: float) -> float:
    q_s = poloidal_mode_number/toroidal_mode_number

    # Note: Chease outputs -d_r, so negate here
    d_r = -np.interp(q_s, chease_cols.q, chease_cols.d_r)
    beta_p = np.interp(q_s, chease_cols.q, chease_cols.beta_p)

    print(d_r)

    return ntm_ggj_term(w, d_r, w_d)

def bootstrap_term(w: float,
                   poloidal_mode_number: float,
                   toroidal_mode_number: float,
                   chease_cols: CheaseColumns,
                   w_d: float) -> float:
    q_s = poloidal_mode_number/toroidal_mode_number
    r_maj = chease_cols.r_avg[0]
    f_val = np.interp(
        q_s,
        chease_cols.q,
        chease_cols.F
    )
    shear_rs = np.interp(
        q_s,
        chease_cols.q,
        chease_cols.shear
    )

    # j_bs_rs = np.interp(
    #     psi_rs,
    #     bootstrap_profile.x_values,
    #     bootstrap_profile.y_values
    # )
    j_bs_rs = np.interp(
        q_s,
        chease_cols.q,
        chease_cols.j_bs
    )

    # Above is in chease units, but the function below
    # requires SI-units. Since the units of
    # R/B^2 * <j.B> cancel, only need to remove a factor
    # of mu0 from <j.B>
    mu0 = 4e-7 * np.pi
    j_bs_rs = j_bs_rs/mu0

    logger.debug(
        "r_maj, f_val, q_s, shear_rs, j_bs_rs", 
        r_maj, f_val, q_s, shear_rs, j_bs_rs
    )
    return ntm_bootstrap_term(
        w, r_maj, f_val, q_s, shear_rs, j_bs_rs, w_d
    )