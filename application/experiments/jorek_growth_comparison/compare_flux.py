from argparse import ArgumentParser
from typing import List
from matplotlib import pyplot as plt
import numpy as np

from jorek_tools.macroscopic_vars_analysis.plot_quantities import MacroscopicQuantity
from jorek_tools.macroscopic_vars_analysis.plot_aligned_energies import jorek_energy_to_dpsi
from tearing_mode_solver.helpers import TimeDependentSolution, load_sim_from_disk

def plot_aligned_solutions(solutions: List[TimeDependentSolution],
                           delta_psi_alignment: float = 1e-12,
                           labels: List[str]=None):
    """
    Align and normalise multiple quasi-linear solutions and
    plot on the same axes.
    """
    fig, ax = plt.subplots(1)

    # If no labels supplied, just create a list of empty strings
    if labels==None:
        labels=[""]*len(solutions)

    for label, sol in zip(labels, solutions):
        delta_psi_norm = sol.psi_t/delta_psi_alignment
        times = sol.times

        t0 = np.interp(
            1.0, delta_psi_norm, times
        )
        times = times - t0

        filt = times > 0.0
        times = times[filt]
        delta_psi_norm = delta_psi_norm[filt]

        ax.plot(times, delta_psi_norm, label=label)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compare delta_psi vs time between JOREK and equivalent model calculation"
    )
    parser.add_argument(
        "ql_solution", help="Path to quasi-linear solution .zip file",
        type=str
    )
    parser.add_argument(
        "jorek_energy", help="Path to JOREK magnetic_energies.dat",
        type=str,
        default="magnetic_energies.dat"
    )
    parser.add_argument(
        "-di", "--delta-psi-init",
        help="Initial delta psi value to align both JOREK and the quasi-linear solution",
        type=float,
        default=1e-12
    )
    args = parser.parse_args()

    jorek_sol = jorek_energy_to_dpsi(
        MacroscopicQuantity(args.jorek_energy),
        args.delta_psi_init
    )
    params, ql_sol = load_sim_from_disk(args.ql_solution)

    plot_aligned_solutions(
        [jorek_sol, ql_sol],
        args.delta_psi_init,
        ["JOREK", "Model"]
    )

    plt.show()


