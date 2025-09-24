from argparse import ArgumentParser
from typing import List
from matplotlib import pyplot as plt
import numpy as np

from jorek_tools.delta_psi_extraction.plot_delta_psi_vs_time import get_psi_vs_time_for_mode
from jorek_tools.macroscopic_vars_analysis.plot_quantities import MacroscopicQuantity
from tearing_mode_solver.helpers import TimeDependentSolution, load_sim_from_disk, TearingModeParameters
from tearing_mode_solver.conversions import solution_time_scale


def plot_aligned_growth_rates(ql_si: TimeDependentSolution,
                              jorek_mag_growth_rates: MacroscopicQuantity,
                              jorek_t0: float):
    """
    Take jorek magnetic growth rate (from magnetic_growth_rates.dat), set
    appropriate t0, and plot against magnetic growth rate from quasi-linear model.

    Note: Ensure you've extracted the SI magnetic growth rates from JOREK.

    :param ql_si: Quasi-linear solution converted to SI
    :param jorek_mag_growth_rates: JOREK magnetic growth rates taken from live data
        Also SI units, though time axis is in ms by default.
    :param jorek_t0: JOREK time at which to align the JOREK solution to the quasi-linear
        solution. Units of ms.
    """
    ql_times = ql_si.times * 1000.0 # Convert to ms
    ql_growth_rates = ql_si.dpsi_dt/ql_si.psi_t

    jorek_times = jorek_mag_growth_rates.x_values
    jorek_growth_rates = jorek_mag_growth_rates.y_values

    jorek_t_filt = jorek_times > jorek_t0
    jorek_times = jorek_times[jorek_t_filt]
    jorek_growth_rates = jorek_growth_rates[jorek_t_filt]

    jorek_times = jorek_times - jorek_t0

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(jorek_times, jorek_growth_rates, label='JOREK', color='black')
    ax.plot(ql_times, ql_growth_rates, label='Model', color='red', linestyle='-')
    
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Growth rate (1/s)")
    ax.legend()
    ax.grid()

    fig.tight_layout()

def plot_aligned_fluxes(ql_si: TimeDependentSolution,
                        jorek_energies: MacroscopicQuantity,
                        jorek_t0: float):
    """
    Take jorek magnetic energies (from magnetic_growth_rates.dat), set
    appropriate t0 where psi0 is known, convert to flux
    and plot against magnetic growth rate from quasi-linear model.

    Note: Ensure you've extracted the SI magnetic energies from JOREK.

    :param ql_si: Quasi-linear solution converted to SI
    :param jorek_energies: JOREK magnetic energies taken from live data
        Also SI units, though time axis is in ms by default.
    :param jorek_t0: JOREK time at which to align the JOREK solution to the quasi-linear
        solution. Units of ms.
    """
    ql_times = ql_si.times * 1000.0 # Convert to ms
    ql_fluxes = ql_si.psi_t

    jorek_times = jorek_energies.x_values
    jorek_energy_vals = jorek_energies.y_values

    jorek_psi_vals = np.sqrt(jorek_energy_vals)

    jorek_t_filt = jorek_times > jorek_t0
    jorek_times = jorek_times[jorek_t_filt]
    jorek_psi_vals = jorek_psi_vals[jorek_t_filt]
    jorek_psi_vals = jorek_psi_vals * ql_fluxes[0]/jorek_psi_vals[0]

    jorek_times = jorek_times - jorek_t0

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(jorek_times, jorek_psi_vals, label='JOREK', color='black')
    ax.plot(ql_times, ql_fluxes, label='Model', color='red', linestyle='-')
    
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Growth rate (1/s)")
    ax.legend()
    ax.grid()

    fig.tight_layout()

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compare delta_psi vs time between JOREK and equivalent model calculation"
    )
    parser.add_argument(
        "-ql", "--ql-solution", help="Path to quasi-linear solution .zip file",
        type=str
    )
    parser.add_argument(
        "-jg", "--jorek-growth-rate", help="Path to jorek magnetic_growth_rates.dat",
        type=str
    )
    parser.add_argument(
        "-je", "--jorek-mag-energies", help="Path to jorek magnetic_energies.dat",
        type=str,
        nargs="?"
    )
    parser.add_argument(
        "-t0", "--initial-time",
        help="Initial time to align the JOREK and quasi-linear solutions, units of ms",
        type=float
    )
    args = parser.parse_args()

    jorek_gr = MacroscopicQuantity(args.jorek_growth_rate)
    jorek_gr.load_x_values_by_index(0)
    jorek_gr.load_y_values_by_index(2)
    params, ql_sol = load_sim_from_disk(args.ql_solution)
    print(params.lundquist_number, params.r_minor, params.rho0)
    ql_sol_si = solution_time_scale(params, ql_sol, True)

    plot_aligned_growth_rates(ql_sol_si, jorek_gr, args.initial_time)

    if args.jorek_mag_energies:
        jorek_energies = MacroscopicQuantity(args.jorek_mag_energies)
        jorek_energies.load_x_values_by_index(0)
        jorek_energies.load_y_values_by_index(2)

        plot_aligned_fluxes(ql_sol_si, jorek_energies, args.initial_time)

    plt.show()


