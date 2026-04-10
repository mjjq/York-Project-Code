from scipy.signal import savgol_filter

from jorek_tools.jorek_dat_to_array import (
    q_and_j_from_csv,
    read_r_minor,
    read_eta_profile_r_minor,
    read_Btor,
    read_R0,
    read_postproc_profiles
)
from tearing_mode_solver.helpers import TearingModeParameters
from tearing_mode_solver.outer_region_solver import (
    eta_to_lundquist_number,
    rational_surface
)
from tearing_mode_solver.profiles import value_at_r
from debug.log import logger

def get_parameters(psi_current_prof_filename: str,
                   q_prof_filename: str,
                   poloidal_mode_number: int,
                   toroidal_mode_number: int,
                   include_bootstrap: bool = False) -> TearingModeParameters:
    q_profile, j_profile = q_and_j_from_csv(
        psi_current_prof_filename, q_prof_filename
    )

    if include_bootstrap:
        try:
            # Column 1 is r_minor. Column 12 is J_bootstrap
            j_bs_profiles = read_postproc_profiles(
                psi_current_prof_filename, 1, 11
            )
            j_bs_prof_t0 = j_bs_profiles[0]
            r_minor = j_bs_prof_t0.x_vals
            # Need to negate the profile as JOREK gives
            # a negative value for some reason
            j_bs = -1.0 * j_bs_prof_t0.y_vals
            j_bs_filt = savgol_filter(j_bs, window_length=11, polyorder=3)

            from matplotlib import pyplot as plt
            plt.plot(rs, j_bs_filt)
            plt.show()

            rs, js = zip(*j_profile)
            js += j_bs_filt

            j_profile = list(zip(rs, js))

        except:
            print("Could not find bootstrap current.")

    r_minor = read_r_minor(psi_current_prof_filename)
    # q_profile is a function of r/r_minor, so multiply by r_minor
    # to get SI
    r_s_si = r_minor*rational_surface(
        q_profile, poloidal_mode_number/toroidal_mode_number
    )

    eta_profile = read_eta_profile_r_minor(psi_current_prof_filename)
    eta_at_rs = value_at_r(eta_profile, r_s_si)
    B_tor = read_Btor(psi_current_prof_filename)
    R_0 = read_R0(psi_current_prof_filename)
    logger.debug(f"{r_minor}, {R_0}, {B_tor}, {eta_at_rs}")
    lundquist_number = eta_to_lundquist_number(
        r_minor,
        R_0,
        B_tor,
        eta_at_rs
    )
    logger.debug(f"Lundquist number: {lundquist_number}")

    params = TearingModeParameters(
        poloidal_mode_number=poloidal_mode_number,
        toroidal_mode_number=toroidal_mode_number,
        lundquist_number=lundquist_number,
        initial_flux=1e-12,
        B0=B_tor,
        R0=R_0,
        q_profile=q_profile,
        j_profile=j_profile,
        r_minor=r_minor
    )
    logger.debug(params)

    return params