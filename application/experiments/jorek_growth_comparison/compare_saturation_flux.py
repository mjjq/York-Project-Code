import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import imports

from jorek_tools.jorek_dat_to_array import q_and_j_from_csv
from tearing_mode_solver.delta_model_solver import solve_time_dependent_system
from tearing_mode_solver.outer_region_solver import \
    solve_system, OuterRegionSolution, normalised_energy_integral, energy, \
    island_width, delta_prime_non_linear
from tearing_mode_solver.helpers import (
    savefig, 
    TearingModeParameters,
    sim_to_disk,
    TimeDependentSolution
)
from jorek_tools.calc_jorek_growth import growth_rate, _name_time, _name_flux
from tearing_mode_solver.profiles import magnetic_shear
from jorek_tools.psi_t_from_vtk import jorek_flux_at_q


def ql_tm_vs_time():
   
    jorek_data_filename = "../../jorek_tools/postproc/psi_t_data.csv"
    psi_current_prof_filename = "../../jorek_tools/postproc/exprs_averaged_s00000.csv"
    q_prof_filename = "../../jorek_tools/postproc/qprofile_s00000.dat"
    q_profile, j_profile = q_and_j_from_csv(
        psi_current_prof_filename, q_prof_filename
    )
    
    params = TearingModeParameters(
        poloidal_mode_number = 2,
        toroidal_mode_number = 1,
        lundquist_number = 4.32e6,
        initial_flux = 1.53e-13,
        B0=1.0,
        R0=40.0,
        q_profile = q_profile,
        j_profile = j_profile
    )
    
    sol = solve_system(params)
    
    fmin = 0.00100
    fmax = 0.00175
    fluxes = np.linspace(fmin, fmax, 100)
    shear_rs = magnetic_shear(params.q_profile, sol.r_s)
    island_widths = island_width(
        fluxes, 
        sol.r_s, 
        params.poloidal_mode_number,
        params.toroidal_mode_number,
        shear_rs
    )
    dps = [sol.r_s * delta_prime_non_linear(sol, w) for w in island_widths]
    
    fig, ax = plt.subplots(1)
    
    ax.hlines(
        0.0, xmin=fmin, xmax=fmax, color='grey', linestyle='--', 
        label='Marginal stability'
    )
    
    ax.plot(fluxes, dps, color='black', label='Model')
    ax.set_xlabel(r"Normalised flux $(a^2 B_{\phi 0})$")
    ax.set_ylabel(r"$r_s \Delta' \left[\delta \psi^{(1)} \right]$")
    
    
    jorek_data = pd.read_csv(jorek_data_filename).fillna(0)
    jorek_times, jorek_flux = jorek_flux_at_q(jorek_data, q_profile, 2/1)
    
    jorek_saturation_flux = jorek_flux[-1]
    
    ax.scatter(
        [jorek_saturation_flux], 0.0, 
        label='JOREK saturation flux', color='red'
    )
    
    ax.legend()
    
    savefig("saturation_flux_comparison")
    
    plt.show()
    
    



if __name__=='__main__':
    ql_tm_vs_time()
    #check_model_t_dependence()
