import numpy as np
from matplotlib import pyplot as plt
from os.path import join, expanduser
import sys

import imports

from jorek_tools.jorek_dat_to_array import q_and_j_from_csv
from tearing_mode_solver.delta_model_solver import solve_time_dependent_system
from tearing_mode_solver.outer_region_solver import (
    solve_system,
    OuterRegionSolution,
    normalised_energy_integral,
    energy,
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

    if len(sys.argv) < 3:
        experiment_root = expanduser(
            "~/csd3/jorek_data/intear_ntor3_cylinder/run_47608261/postproc"
        )

        psi_current_prof_filename = join(experiment_root, "exprs_averaged_s00000.csv")
        q_prof_filename = join(experiment_root, "qprofile_s00000.dat")
    else:
        psi_current_prof_filename = sys.argv[1]
        q_prof_filename = sys.argv[2]

    q_profile, j_profile = q_and_j_from_csv(psi_current_prof_filename, q_prof_filename)

    # rq, q = zip(*q_profile)
    # q = np.array(q)/q[0]
    # rj, js = zip(*j_profile)
    # js = 10.0*np.array(js)
    # j_profile = list(zip(rj, 40.0*np.array(js)))
    # fig, ax = plt.subplots(3)
    # ax[0].plot(rq, q)
    # ax[1].plot(rj, js)
    # ax[2].plot(rj, dj_dr_vals)

    params = TearingModeParameters(
        poloidal_mode_number=2,
        toroidal_mode_number=1,
        lundquist_number=1.119e10,
        initial_flux=1.336e-12,
        B0=1.0,
        R0=40.0,
        q_profile=q_profile,
        j_profile=j_profile,
    )

    sol = solve_system(params)

    t0 = 4.2444e5  # This is the jorek time at which the simulation numerically stabilises

    times = np.linspace(t0, 2e9, 200000)
    print(max(times))

    ql_solution = solve_time_dependent_system(params, times)
    times = ql_solution.times
    psi_t = ql_solution.psi_t
    dpsi_t = ql_solution.dpsi_dt
    w_t = ql_solution.w_t

    print(times)
    print(ql_solution.psi_t)
    # plt.plot(w_t)
    # plt.show()

    # print("lgr: ", lin_growth_rate)
    # print("Threshold: ", ql_threshold)
    # print(psi_t)

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

    fname = f"jorek_model_(m,n)=({params.poloidal_mode_number},{params.toroidal_mode_number})"
    savefig(fname)
    sim_to_disk(fname, params, ql_solution)

    plot_growth(times, dpsi_t, psi_t)
    plot_delta_prime(sol, ql_solution)
    plot_energy(params, ql_solution, sol)

    plt.show()


if __name__ == "__main__":
    ql_tm_vs_time()
    # plot_input_profiles()
