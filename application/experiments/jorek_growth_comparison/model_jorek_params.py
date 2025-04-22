import numpy as np
from os.path import join, expanduser
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt

#import imports

from jorek_tools.jorek_dat_to_array import (
    q_and_j_from_csv, 
    read_eta_profile_r_minor,
    read_Btor,
    read_R0,
    read_r_minor
)
from tearing_mode_solver.delta_model_solver import solve_time_dependent_system
from tearing_mode_solver.outer_region_solver import (
    solve_system,
    OuterRegionSolution,
    normalised_energy_integral,
    energy,
    rational_surface,
    eta_to_lundquist_number
)
from tearing_mode_solver.helpers import (
    savefig,
    TearingModeParameters,
    sim_to_disk,
    TimeDependentSolution,
)
from jorek_tools.time_conversion import jorek_to_alfven_time

def plot_growth(times, dpsi_t, psi_t):
    fig_growth, ax_growth = plt.subplots(1, figsize=(4.5, 3))

    ax_growth.plot(times, dpsi_t / psi_t, color="black")
    ax_growth.set_ylabel(r"Growth rate $\delta\dot{\psi}^{(1)}/\delta\psi^{(1)}$")
    ax_growth.set_xlabel(r"Normalised time $1/\bar{\omega}_A$")

    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    fig_growth.tight_layout()

    ax_growth.set_xscale("log")
    ax_growth.set_yscale("log")

    ax_growth.plot(times, 2 / times, color="red", label="2/t dependence")
    # orig_fname, ext = os.path.splitext(os.path.basename(model_data_filename))
    # savefig("jorek_model_growth_rate")


def plot_energy(
    params: TearingModeParameters,
    td_sol: TimeDependentSolution,
    outer_sol: OuterRegionSolution,
):
    norm_energy = normalised_energy_integral(outer_sol, params)

    print(f"Normalised energy integral: {norm_energy}")

    energies = energy(td_sol.psi_t, params, norm_energy)

    fig, axs = plt.subplots(2)

    ax, ax2 = axs

    ax.plot(td_sol.times, energies)

    ax.set_yscale("log")
    ax.set_xlabel("Time ($1/\omega_A$)")
    ax.set_ylabel("Energy")

    ax2.plot(outer_sol.r_range_fwd, outer_sol.psi_forwards)
    ax2.plot(outer_sol.r_range_bkwd, outer_sol.psi_backwards)
    ax2.set_xlabel(r"Minor radial co-ordinate ($r/a$)")
    ax2.set_ylabel(r"Perturbed flux $(a^2 B_{\phi 0})$")

    fig.tight_layout()


def plot_outer_region_solution(params: TearingModeParameters):
    tm = solve_system(params)

    fig, axs = plt.subplots(2)
    ax, ax_dpsi_dr = axs

    max_psi = max(np.max(tm.psi_forwards), np.max(tm.psi_backwards))
    ax.plot(tm.r_range_fwd, tm.psi_forwards / max_psi)
    ax.plot(tm.r_range_bkwd, tm.psi_backwards / max_psi)

    ax.set_ylabel("Normalised perturbed flux")

    ax_dpsi_dr.plot(tm.r_range_fwd, tm.dpsi_dr_forwards)
    ax_dpsi_dr.plot(tm.r_range_bkwd, tm.dpsi_dr_backwards)

    ax_dpsi_dr.set_ylabel("$\partial \delta\psi^{(1)}/\partial r$")

    ax_dpsi_dr.set_xlabel("Normalised minor radial co-ordinate (a)")

    fname = f"jorek_model_growth_(m,n)=({params.poloidal_mode_number},{params.toroidal_mode_number})"
    # savefig(fname)


def plot_delta_prime(
    outer_sol: OuterRegionSolution, time_dep_sol: TimeDependentSolution
):
    r_s = outer_sol.r_s
    dps = r_s * np.array(time_dep_sol.delta_primes)
    times = time_dep_sol.times

    fig, ax = plt.subplots(1)
    ax.plot(times, dps)

    ax.set_xlabel("Normalised time (1/$\omega_A$)")
    ax.set_ylabel("$r_s\Delta'(w)$")

    fig.tight_layout()


def plot_input_profiles():
    psi_current_prof_filename = "../../jorek_tools/postproc/exprs_averaged_s00000.csv"
    q_prof_filename = "../../jorek_tools/postproc/qprofile_s00000.dat"
    q_profile, j_profile = q_and_j_from_csv(psi_current_prof_filename, q_prof_filename)

    fig, axs = plt.subplots(2)

    axq, axj = axs

    rs, qs = zip(*q_profile)
    axq.plot(rs, qs)
    # axq.set_xlabel('Minor radial coordinate (a)')
    axq.set_ylabel("Safety factor")

    rs, js = zip(*j_profile)
    axj.plot(rs, js)
    axj.set_xlabel("Minor radial coordinate (a)")
    axj.set_ylabel("Current density")

    plt.show()


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
        help="Final simulation time"
    )
    parser.add_argument(
        '-s', '--n-steps', type=int, default=10000, 
        help='Number of simulation steps'
    )
    parser.add_argument(
        '-p', '--plot-result', action='store_true',
        help="Choose whether to plot the results"
    )
    args = parser.parse_args()

    psi_current_prof_filename = args.exprs_averaged
    q_prof_filename = args.q_profile

    q_profile, j_profile = q_and_j_from_csv(psi_current_prof_filename, q_prof_filename)

    poloidal_mode_number = args.poloidal_mode_number
    toroidal_mode_number = args.toroidal_mode_number
    init_flux = args.psi_init # JOREK flux at which the simulation numerically stabilises
    t0 = args.initial_time  # This is the jorek time at which the simulation numerically stabilises
    nsteps = args.n_steps

    r_minor = read_r_minor(psi_current_prof_filename)
    # q_profile is a function of r/r_minor, so multiply by r_minor
    # to get SI
    r_s_si = r_minor*rational_surface(q_profile, poloidal_mode_number/toroidal_mode_number)

    eta_profile = read_eta_profile_r_minor(psi_current_prof_filename)
    r_vals, eta_vals = zip(*eta_profile)
    eta_at_rs = np.interp(r_s_si, r_vals, eta_vals)
    B_tor = read_Btor(psi_current_prof_filename)
    lundquist_number = eta_to_lundquist_number(
        r_s_si,
        B_tor,
        eta_at_rs
    )
    R_0 = read_R0(psi_current_prof_filename)

    params = TearingModeParameters(
        poloidal_mode_number=poloidal_mode_number,
        toroidal_mode_number=toroidal_mode_number,
        lundquist_number=lundquist_number,
        initial_flux=init_flux,
        B0=B_tor,
        R0=R_0,
        q_profile=q_profile,
        j_profile=j_profile,
    )

    sol = solve_system(params)

    # Typically, several lundquist numbers needed to reach saturation
    t1 = t0 + 2.0*lundquist_number
    if args.final_time:
        t1 = args.final_time

    times = np.linspace(t0, t1, nsteps)
    print(max(times))

    ql_solution = solve_time_dependent_system(params, times)
    times = ql_solution.times
    psi_t = ql_solution.psi_t
    dpsi_t = ql_solution.dpsi_dt
    w_t = ql_solution.w_t

    #print(times)
    #print(ql_solution.psi_t)
    # plt.plot(w_t)
    # plt.show()

    # print("lgr: ", lin_growth_rate)
    # print("Threshold: ", ql_threshold)
    # print(psi_t)
    
    fname = f"jorek_model_m{params.poloidal_mode_number}_n{params.toroidal_mode_number}"
    sim_to_disk(fname, params, ql_solution)

    if args.plot_result:

        fig, ax = plt.subplots(1, figsize=(4, 3.5))
        ax2 = ax.twinx()

        ax.plot(times, psi_t, label="Flux", color="black")

        ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
        ax.set_ylabel(r"Normalised perturbed flux ($\delta \psi^{(1)}$)")

        ax2.plot(times, w_t, label="Normalised island width", color="red")
        ax2.set_ylabel(r"Normalised electrostatic modal width ($\hat{\delta}_{ql}$)")
        ax2.yaxis.label.set_color("red")

        ax.legend(prop={"size": 7}, loc="lower right")

        # ax.set_yscale('log')
        # ax2.set_yscale('log')
        ax.set_xscale("log")
        ax2.set_xscale("log")
        ax.set_yscale("log")
        ax2.set_yscale("log")

        fig.tight_layout()
        # plt.show()

        savefig(fname)

        plot_growth(times, dpsi_t, psi_t)
        plot_delta_prime(sol, ql_solution)
        plot_energy(params, ql_solution, sol)

        plt.show()


if __name__ == "__main__":
    ql_tm_vs_time()
    # plot_input_profiles()
