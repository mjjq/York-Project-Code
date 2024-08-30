import numpy as np
from matplotlib import pyplot as plt

import imports
from tearing_mode_solver.outer_region_solver import rational_surface, magnetic_shear
from tearing_mode_solver.backward_rutherford_solver import solve_time_dependent_system
from tearing_mode_solver.helpers import (
    classFromArgs,
    TimeDependentSolution,
    savefig,
    load_sim_from_disk,
    TearingModeParameters
)
from tearing_mode_solver.outer_region_solver import island_width
from tearing_mode_solver.algebraic_fitting import get_parab_coefs
from jorek_tools.calc_jorek_growth import growth_rate, _name_time, _name_flux
from jorek_tools.jorek_dat_to_array import q_and_j_from_csv
from jorek_tools.psi_t_from_vtk import jorek_flux_at_q
from jorek_tools.time_conversion import jorek_to_alfven_time


if __name__ == '__main__':
    model_data_filename = "./output/05-06-2024_16:42_jorek_model_(m,n)=(2,1).zip"
    t_range = np.linspace(0.0, 2e9, 100000)
    
    params, ql_sol = load_sim_from_disk(model_data_filename)
    
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

    plt.show()
