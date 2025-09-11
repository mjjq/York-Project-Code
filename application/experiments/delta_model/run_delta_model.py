import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt

from tearing_mode_solver.profiles import generate_j_profile, generate_q_profile
from tearing_mode_solver.delta_model_solver import solve_time_dependent_system
from tearing_mode_solver.outer_region_solver import (
    solve_system,
    OuterRegionSolution,
    growth_rate,
    delta_prime_non_linear
)
from tearing_mode_solver.helpers import (
    TearingModeParameters,
    sim_to_disk,
    TimeDependentSolution,
)
from tearing_mode_plotter.plot_magnetic_island_width import phase_plot, island_width_plot
from tearing_mode_plotter.plot_quasi_linear_solution import plot_perturbed_flux, plot_growth_rate
from jorek_tools.quasi_linear_model.central_density_si import central_density_si


def ql_tm_vs_time():
    """
    Numerically solve the quasi-linear time-dependent tearing mode problem
    and plot the perturbed flux and layer width as functions of time.
    """
    parser = ArgumentParser(
		prog="python3 model_jorek_params.py",
		description="Solves quasi-linear tearing mode equations" \
			" to describe tearing mode growth as a function of time. "\
            "Automatically output data into .zip format. ",
        epilog="Run this script in the `postproc` folder of the simulation " \
            "run to avoid locating exprs_averaged and qprofile files " \
            "manually. Need to run ./jorek2_postproc < get_flux.pp first.",
        formatter_class=ArgumentDefaultsHelpFormatter
	)
    parser.add_argument(
        '-lq', '--lundquist-number', type=float,
        help="Lundquist number at the rational surface",
        default=1e6
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
        '-b0', '--toroidal-field', type=float, default=1.0,
        help="Toroidal field strength at magnetic axis (T)"
    )
    parser.add_argument(
        '-r0', '--major-radius', type=float, default=10.0,
        help="Tokamak major radius (m)"
    )
    parser.add_argument(
        '-p0', '--psi-init', type=float, default=1e-12, 
        help="Default initial perturbed flux"
    )
    parser.add_argument(
        '-t0', '--initial-time', type=float, default=4.2444e5,
        help="Initial simulation time (to align with JOREK initial)"
    )
    parser.add_argument(
        '-t1', 
        '--final-time', 
        type=float,
        help="Final simulation time (in units of lundquist number).",
	default=0.1
    )
    parser.add_argument(
        '-s', '--n-steps', type=int, default=10000, 
        help='Number of simulation steps'
    )
    parser.add_argument(
        '-p', '--plot-result', action='store_true',
        help="Choose whether to plot the results"
    )
    parser.add_argument(
        '-si', '--si-time', action='store_true',
        help="Choose whether to plot time in SI units " \
        "If left false, time is given in terms of Alfven time" \
        "Note, data is still saved in terms of Alfven time"
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

    #psi_current_prof_filename = args.exprs_averaged
    #q_prof_filename = args.q_profile

    #q_profile, j_profile = q_and_j_from_csv(psi_current_prof_filename, q_prof_filename)

    q_profile = generate_q_profile(axis_q=1.0, shaping_exponent=2.0)
    j_profile = generate_j_profile(axis_q=1.0, shaping_exponent=2.0)

    poloidal_mode_number = args.poloidal_mode_number
    toroidal_mode_number = args.toroidal_mode_number
    init_flux = args.psi_init # JOREK flux at which the simulation numerically stabilises
    t0 = args.initial_time  # This is the jorek time at which the simulation numerically stabilises
    nsteps = args.n_steps

    lundquist_number = args.lundquist_number
    print(f"lundquist number: {lundquist_number:.2g}")

    rho0 = central_density_si(
        args.central_mass,
        args.central_density
    )
    B_tor = args.toroidal_field
    R_0 = args.major_radius

    params = TearingModeParameters(
        poloidal_mode_number=poloidal_mode_number,
        toroidal_mode_number=toroidal_mode_number,
        lundquist_number=lundquist_number,
        initial_flux=init_flux,
        B0=B_tor,
        R0=R_0,
        q_profile=q_profile,
        j_profile=j_profile,
        rho0=rho0
    )

    # Typically, several lundquist numbers needed to reach saturation
    t1 = t0 + 0.1*lundquist_number
    if args.final_time:
        t1 = t0 + args.final_time*lundquist_number

    times = np.linspace(t0, t1, nsteps)
    #print(max(times))

    outer_sol = solve_system(params)
    delta_p, gr = growth_rate(
        params.poloidal_mode_number,
        params.toroidal_mode_number,
        lundquist_number,
        q_profile,
        outer_sol
    )

    print(f"Delta'(0)={delta_p:.2g}")

    ql_solution = solve_time_dependent_system(params, times)
    
    fname = f"jorek_model_m{params.poloidal_mode_number}_n{params.toroidal_mode_number}"
    sim_to_disk(fname, params, ql_solution)

    if args.plot_result:
        phase_plot(params, ql_solution, args.si_time)
        plot_perturbed_flux(params, ql_solution, args.si_time)
        plot_growth_rate(params, ql_solution, args.si_time)
        island_width_plot(params, ql_solution, args.si_time)

        plt.show()


if __name__ == "__main__":
    ql_tm_vs_time()
    # plot_input_profiles()
