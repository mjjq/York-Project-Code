import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from debug.log import logger

from jorek_tools.jorek_dat_to_array import (
    q_and_j_from_csv, 
    read_eta_profile_r_minor,
    read_Btor,
    read_R0,
    read_r_minor,
    read_chi_par_profile_rminor,
    read_chi_perp_profile_rminor
)
from tearing_mode_solver.profiles import value_at_r
from tearing_mode_solver.outer_region_solver import (
    solve_system,
    rational_surface,
    eta_to_lundquist_number,
    delta_prime,
    diffusion_width,
    curvature_stabilisation,
    magnetic_shear,
    growth_rate_full,
    alfven_frequency
)
from tearing_mode_solver.helpers import (
    TearingModeParameters
)

if __name__=='__main__':
    parser = ArgumentParser(
		description="""Solve linear tearing mode outer solution, calculate
        Delta' and curvature stabilisation term, then calculate effective
        linear growth rate""",
        epilog="Run this script in the `postproc` folder of the simulation " \
            "run to avoid locating exprs_averaged and qprofile files " \
            "manually. Need to run ./jorek2_postproc < get_flux.pp first.",
        formatter_class=ArgumentDefaultsHelpFormatter
	)
    parser.add_argument(
        'resistive_interchange', nargs='+', type=float,
        help="List of resistive interchange values"
    )
    parser.add_argument(
        '-ex', '--exprs-averaged',  type=str, default="exprs_averaged_s00000.dat",
        help="Path to exprs_averaged...dat postproc file (Optional)"
    )
    parser.add_argument(
        '-q', '--q-profile', type=str, default="qprofile_s00000.dat",
        help="Path to qprofile...dat file (Optional)"
    )
    parser.add_argument(
        '-m', '--poloidal-mode-number', type=int, default=2,
        help="Poloidal mode number of the tearing mode"
    )
    parser.add_argument(
        '-n', '--toroidal-mode-number', type=int, default=1,
        help="Toroidal mode number of the tearing mode"
    )
    parser.add_argument(
        '-cm', '--central-mass', type=float,
        help="Central mass (as per JOREK namelist, unitless)",
        default=2.0
    )
    parser.add_argument(
        '-cd', '--central-density', type=float,
        help="Central number density of plasma (10^20/m^3)",
        default=1.0
    )

    args = parser.parse_args()

    psi_current_prof_filename = args.exprs_averaged
    q_prof_filename = args.q_profile

    q_profile, j_profile = q_and_j_from_csv(
        psi_current_prof_filename, q_prof_filename
    )

    poloidal_mode_number = args.poloidal_mode_number
    toroidal_mode_number = args.toroidal_mode_number

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
        initial_flux=0.0, # Not solving time-dependent system, don't care what this is
        B0=B_tor,
        R0=R_0,
        q_profile=q_profile,
        j_profile=j_profile,
    )
    logger.debug(params)

    outer_solution = solve_system(params)

    delta_p = delta_prime(outer_solution)


    chi_perp_profile = read_chi_perp_profile_rminor(args.exprs_averaged)
    chi_perp_rs = value_at_r(chi_perp_profile, r_s_si)

    chi_par_profile = read_chi_par_profile_rminor(args.exprs_averaged)
    chi_par_rs = value_at_r(chi_par_profile, r_s_si)

    mag_shear = magnetic_shear(q_profile, outer_solution.r_s)
    diff_width = diffusion_width(
        chi_perp_rs,
        chi_par_rs,
        outer_solution.r_s,
        R_0/r_minor,
        params.toroidal_mode_number,
        mag_shear
    )

    resistive_interchange_values = args.resistive_interchange

    m_proton = 1.6726e-27
    # See https://www.jorek.eu/wiki/doku.php?id=normalization
    n0_normalisation = 1e20
    rho0 = args.central_mass * args.central_density * n0_normalisation * m_proton

    alfven_freq = alfven_frequency(R_0, B_tor, rho0)

    for d_r in resistive_interchange_values:
        curv_stabilisation = curvature_stabilisation(diff_width, d_r)
        delta_p_eff = delta_p + curv_stabilisation

        gr = growth_rate_full(
            poloidal_mode_number,
            toroidal_mode_number,
            lundquist_number,
            outer_solution.r_s,
            mag_shear,
            delta_p_eff
        )

        print(gr*alfven_freq)