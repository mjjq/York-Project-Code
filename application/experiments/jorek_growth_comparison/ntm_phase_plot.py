import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt

from jorek_tools.quasi_linear_model.get_tm_parameters import (
    get_parameters
)
from jorek_tools.quasi_linear_model.get_diffusion_width import (
    get_diffusion_width
)
from tearing_mode_solver.outer_region_solver import (
    solve_system,
    delta_prime_non_linear,
    curvature_stabilisation_non_linear,
)

if __name__=='__main__':
    parser = ArgumentParser(
		description="""Solve linear tearing mode outer solution, calculate
        Delta' in both linear and non-linear regimes, plot as a function of w""",
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
    parser.add_argument(
        '-wd', '--diffusion-width', nargs='+', type=float,
        help="Custom array of diffusion widths",
        default=[]
    )

    args = parser.parse_args()

    params = get_parameters(
        args.exprs_averaged,
        args.q_profile,
        args.poloidal_mode_number,
        args.toroidal_mode_number
    )

    outer_solution = solve_system(params)

    if args.diffusion_width:
        diff_width = args.diffusion_width
    else:
        # Must be in array format
        diff_width = [get_diffusion_width(
            args.exprs_averaged,
            args.q_profile,
            args.poloidal_mode_number,
            args.toroidal_mode_number
        )]

    # m_proton = 1.6726e-27
    # # See https://www.jorek.eu/wiki/doku.php?id=normalization
    # n0_normalisation = 1e20
    # rho0 = args.central_mass * args.central_density * n0_normalisation * m_proton

    # gr_conversion = 1.0
    # if args.si_units:
    #     gr_conversion = alfven_frequency(params.R0, params.B0, rho0)

    fig, ax = plt.subplots(1)
    w_range = np.linspace(0.0, 0.5, 1000)
    ax.hlines(
        0.0, np.min(w_range), np.max(w_range), 
        'black', linestyles='--'
    )
    ax.grid()

    for w_d in diff_width:
        for d_r in args.resistive_interchange:
            delta_ps_classical = np.array([delta_prime_non_linear(
                outer_solution, 
                w
            ) for w in w_range])
            delta_ps_curv = np.array([curvature_stabilisation_non_linear(
                w_d, 
                d_r, 
                w
            ) for w in w_range])
            delta_ps_eff = delta_ps_classical + delta_ps_curv

            ax.plot(w_range, delta_ps_eff, label=f'$D_R={d_r:.3f}, w_d/a={w_d:.4f}$')

    ax.legend()
    ax.set_xlabel("w/a")
    ax.set_ylabel("$\\tau_r/1.22\, d(w/a)/dt$")

    plt.show()
