import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt

from jorek_tools.quasi_linear_model.central_density_si import central_density_si
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
    alfven_frequency
)
from tearing_mode_solver.helpers import load_sim_from_disk

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
        "jorek_ntm_files", nargs='+', type=str,
        help="List of JOREK extracted magnetic island data files"\
        "(See ntm_phase_jorek_data.py)",
        default=[]
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
        '-wd', '--diffusion-widths', nargs='+', type=float,
        help="List of diffusion widths (to label JOREK data)"
    )

    args = parser.parse_args()

    fig, ax = plt.subplots(1)
    ax.grid()

    w_d_vals = args.diffusion_widths

    for w_d, jorek_data_filename in zip(w_d_vals, args.jorek_ntm_files):
        params, jorek_island_data = load_sim_from_disk(jorek_data_filename)

        rho0 = central_density_si(args.central_mass, args.central_density)

        tau_R = params.lundquist_number / alfven_frequency(
            params.R0, 
            params.B0, 
            rho0
        )

        w_vals = jorek_island_data.w_t
        t_vals = jorek_island_data.times

        dw = np.diff(w_vals)
        dt = np.diff(t_vals)

        dwdt = (tau_R /1.22) * dw/dt

        ax.plot(w_vals[:-1], dwdt, label=f'$w_d/a={w_d:.3f}$')

    ax.legend()
    ax.set_xlabel("w/a")
    ax.set_ylabel(r"$\tau_R/1.22 \cdot d(w/a)/dt$")

    plt.show()
