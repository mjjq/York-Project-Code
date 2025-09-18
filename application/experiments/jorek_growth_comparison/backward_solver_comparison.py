import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser

from tearing_mode_solver.backward_rutherford_solver import solve_time_dependent_system
from tearing_mode_solver.helpers import (
    TimeDependentSolution,
    savefig,
    load_sim_from_disk,
    TearingModeParameters
)


def plot_against_backward_rutherford(params: TearingModeParameters,
                                     ql_sol: TimeDependentSolution):
    t_range = np.linspace(0.0, ql_sol.times[-1], 100000)
    
    params.initial_flux = ql_sol.psi_t[-1]
    initial_time = ql_sol.times[-1]
    
    td_sol, outer_sol = solve_time_dependent_system(params, t_range)

    fig, ax = plt.subplots(1, figsize=(5,4))
    
    ax.plot(
        ql_sol.times, 
        ql_sol.w_t, 
        color='black', 
        label='Quasi-linear solution'
    )
    ax.plot(
        initial_time - td_sol.times, 
        td_sol.w_t, 
        color='red', 
        linestyle='--', 
        label='Backward RE'
    )


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left=1e5)
    
    ax.set_xlabel(r"Time ($\omega_A$)")
    ax.set_ylabel(r"Magnetic island width ($a$)")
    
    ax.legend()
    
    fig.tight_layout()

    savefig("backward_rutherford_solution")

if __name__ == '__main__':
    parser = ArgumentParser(
        description="Load a quasi-linear solution and compare it "
        "against the Rutherford equation solved in reverse"
    )
    parser.add_argument(
        "quasilinear_filename", type=str,
        help="Path to quasi-linear solution .zip file."
    )
    args = parser.parse_args()

    model_data_filename = args.quasilinear_filename

    params, ql_sol = load_sim_from_disk(model_data_filename)

    plot_against_backward_rutherford(params, ql_sol)

    plt.show()
