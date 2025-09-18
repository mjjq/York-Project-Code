import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser

from tearing_mode_solver.backward_rutherford_solver import solve_time_dependent_system
from tearing_mode_solver.linear_solver import solve_time_dependent_system as solve_linear
from tearing_mode_solver.helpers import (
    TimeDependentSolution,
    savefig,
    load_sim_from_disk,
    TearingModeParameters
)
from tearing_mode_solver.outer_region_solver import layer_width, growth_rate


def plot_against_backward_rutherford(params: TearingModeParameters,
                                     ql_sol: TimeDependentSolution):
    t_range = np.linspace(0.0, ql_sol.times[-1], 20000)
    
    params.initial_flux = ql_sol.psi_t[-1]
    initial_time = ql_sol.times[-1]
    
    td_sol, outer_sol = solve_time_dependent_system(params, t_range)

    params.initial_flux = ql_sol.psi_t[0]
    exp_sol, outer_sol_2 = solve_linear(params, t_range)
    print(exp_sol.psi_t)
    print(exp_sol.times)
    print(exp_sol.w_t)

    filt = exp_sol.psi_t < max(ql_sol.psi_t)
    exp_w_t = exp_sol.w_t[filt]
    print(exp_w_t)
    exp_times = exp_sol.times[filt]

    resistive_layer_width = layer_width(params)

    fig, ax = plt.subplots(1, figsize=(5,4))
    
    ax.plot(
        ql_sol.times, 
        ql_sol.w_t, 
        color='black', 
        label='Quasi-linear solution'
    )
    ax.plot(
        exp_times,
        exp_w_t,
        color='orange',
        linestyle='--',
        label='Linear solution'
    )
    ax.plot(
        initial_time - td_sol.times, 
        td_sol.w_t, 
        color='red', 
        linestyle='--', 
        label='Rutherford'
    )
    ax.hlines(
        resistive_layer_width, 
        min(td_sol.times), 
        max(td_sol.times),
        label="Resistive layer width"
    )
    

    ax.set_title(f"S={params.lundquist_number:.2g}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel(r"Time ($\omega_A$)")
    ax.set_ylabel(r"Magnetic island width ($a$)")
    
    ax.legend()
    ax.grid(which='major')
    ax.grid(which='minor', alpha=0.1)
    
    fig.tight_layout()


    # fig2, ax2 = plt.subplots(1)

    # ql_dw_dt = np.diff(ql_sol.w_t)/np.diff(ql_sol.times)
    # ax2.plot(
    #     ql_sol.delta_primes[:-1], 
    #     ql_dw_dt, 
    #     color='black',
    #     label="Quasi-linear solution"
    # )

    # ruth_dw_dt = np.diff(td_sol.w_t)/np.diff(initial_time-td_sol.times)
    # ax2.plot(
    #     td_sol.delta_primes[:-1],
    #     ruth_dw_dt,
    #     color='red',
    #     linestyle='--',
    #     label="Backward RE"
    # )





if __name__ == '__main__':
    parser = ArgumentParser(
        description="Load a quasi-linear solution and compare it "
        "against the Rutherford equation solved in reverse"
    )
    parser.add_argument(
        "quasilinear_filename", type=str,
        nargs='+',
        help="Path to quasi-linear solution .zip file."
    )
    args = parser.parse_args()

    model_data_filenames = args.quasilinear_filename

    for model_data_filename in model_data_filenames:
        params, ql_sol = load_sim_from_disk(model_data_filename)
        plot_against_backward_rutherford(params, ql_sol)

    plt.show()
