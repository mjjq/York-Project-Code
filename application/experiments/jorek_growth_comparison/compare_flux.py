from argparse import ArgumentParser
from typing import List
from matplotlib import pyplot as plt
import numpy as np

from jorek_tools.delta_psi_extraction.plot_delta_psi_vs_time import get_psi_vs_time_for_mode
from tearing_mode_solver.helpers import TimeDependentSolution, load_sim_from_disk
from tearing_mode_solver.conversions import solution_time_scale

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
        delta_psi_norm = sol.psi_t
        times = sol.times

        t0 = np.interp(
            delta_psi_alignment, delta_psi_norm, times
        )
        times = times - t0

        filt = times > 0.0
        times = times[filt]
        delta_psi_norm = delta_psi_norm[filt]

        ax.plot(times, delta_psi_norm, label=label)

    ax.grid()
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel(r"$\delta\psi$ (Tm$^2$)")

    ax.legend()

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
        "-jf", "--jorek-fourier-files", help="List of four2d .dat filenames",
        type=str,
        nargs="+"
    )
    parser.add_argument(
        "-qp", "--q-profile", help="Path to qprofile dat file",
        type=str
    )
    parser.add_argument(
        "-tm", "--timestep-map", help="Path to SI timestep map",
        type=str
    )
    parser.add_argument(
        "-di", "--delta-psi-init",
        help="Initial delta psi value to align both JOREK and the quasi-linear solution",
        type=float,
        default=1e-12
    )
    args = parser.parse_args()

    jorek_sol = get_psi_vs_time_for_mode(
        args.jorek_fourier_files,
        [2], 1,
        args.q_profile,
        args.timestep_map
    )[0]
    params, ql_sol = load_sim_from_disk(args.ql_solution)
    ql_sol_si = solution_time_scale(params, ql_sol, True)

    plot_aligned_solutions(
        [jorek_sol, ql_sol_si],
        args.delta_psi_init,
        ["JOREK", "Model"]
    )

    plt.show()


