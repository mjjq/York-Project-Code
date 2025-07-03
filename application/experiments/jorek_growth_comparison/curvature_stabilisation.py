import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from debug.log import logger

from jorek_tools.jorek_dat_to_array import (
    read_r_minor,
    read_chi_par_profile_rminor,
    read_chi_perp_profile_rminor
)
from jorek_tools.quasi_linear_model.get_tm_parameters import get_parameters
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
    parser.add_argument(
        '-si', '--si-units', action='store_true',
        help="Enable this flag to print with SI units. Otherwise, "\
        "results are printed normalised to Alfven frequency",
        default=False
    )

    args = parser.parse_args()

    params = get_parameters(
        args.exprs_averaged,
        args.q_profile,
        args.poloidal_mode_number,
        args.toroidal_mode_number
    )

    outer_solution = solve_system(params)

    delta_p = delta_prime(outer_solution)

    r_minor = read_r_minor(args.exprs_averaged)
    r_s_si = r_minor*rational_surface(
        params.q_profile, 
        params.poloidal_mode_number/params.toroidal_mode_number
    )

    chi_perp_profile = read_chi_perp_profile_rminor(args.exprs_averaged)
    chi_perp_rs = value_at_r(chi_perp_profile, r_s_si)

    chi_par_profile = read_chi_par_profile_rminor(args.exprs_averaged)
    chi_par_rs = value_at_r(chi_par_profile, r_s_si)

    mag_shear = magnetic_shear(params.q_profile, outer_solution.r_s)
    diff_width = diffusion_width(
        chi_perp_rs,
        chi_par_rs,
        outer_solution.r_s,
        params.R0/r_minor,
        params.toroidal_mode_number,
        mag_shear
    )

    resistive_interchange_values = args.resistive_interchange

    m_proton = 1.6726e-27
    # See https://www.jorek.eu/wiki/doku.php?id=normalization
    n0_normalisation = 1e20
    rho0 = args.central_mass * args.central_density * n0_normalisation * m_proton

    gr_conversion = 1.0
    if args.si_units:
        gr_conversion = alfven_frequency(params.R0, params.B0, rho0)

    for d_r in resistive_interchange_values:
        curv_stabilisation = curvature_stabilisation(
            params.lundquist_number,
            d_r,
            mag_shear,
            params.poloidal_mode_number,
            params.toroidal_mode_number,
            outer_solution.r_s
        )
        delta_p_eff = delta_p + curv_stabilisation

        gr = growth_rate_full(
            params.poloidal_mode_number,
            params.toroidal_mode_number,
            params.lundquist_number,
            outer_solution.r_s,
            mag_shear,
            delta_p_eff
        )

        print(gr*gr_conversion)
