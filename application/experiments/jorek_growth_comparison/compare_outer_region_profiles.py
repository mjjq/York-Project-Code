import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import UnivariateSpline

import imports

from jorek_tools.jorek_dat_to_array import q_and_j_from_csv
from tearing_mode_solver.delta_model_solver import solve_time_dependent_system
from tearing_mode_solver.outer_region_solver import (
    solve_system, normalised_energy_integral, energy
)
from tearing_mode_solver.helpers import (
    savefig, 
    TearingModeParameters,
    sim_to_disk
)

def plot_growth(times, dpsi_t, psi_t):
    fig_growth, ax_growth = plt.subplots(1, figsize=(4.5,3))

    ax_growth.plot(times, dpsi_t/psi_t, color='black')
    ax_growth.set_ylabel(r'Growth rate $\delta\dot{\psi}^{(1)}/\delta\psi^{(1)}$')
    ax_growth.set_xlabel(r'Normalised time $1/\bar{\omega}_A$')

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig_growth.tight_layout()

    ax_growth.set_xscale('log')
    #orig_fname, ext = os.path.splitext(os.path.basename(model_data_filename))
    #savefig(f"{orig_fname}_growth_rate")

def plot_outer_region_solution(params: TearingModeParameters, 
                               jorek_psi_data: pd.DataFrame):
    tm = solve_system(params)
    
    fig, axs = plt.subplots(2)
    ax, ax_dpsi_dr = axs
    
    max_psi = max(
        np.max(tm.psi_forwards), 
        np.max(tm.psi_backwards)
    )
    ax.plot(
        tm.r_range_fwd, tm.psi_forwards/max_psi, color='black', label='Model'
    )
    ax.plot(
        tm.r_range_bkwd, tm.psi_backwards/max_psi, color='black'
    )
    
   
    ax_dpsi_dr.plot(
        tm.r_range_fwd, tm.dpsi_dr_forwards/max_psi, color='black'
    )
    ax_dpsi_dr.plot(
        tm.r_range_bkwd, tm.dpsi_dr_backwards/max_psi, color='black'
    )
    
    
    jorek_rs = jorek_psi_data['arc_length']
    jorek_psi = jorek_psi_data['Psi']
    ax.plot(
        jorek_rs, jorek_psi/max(jorek_psi), color='red', 
        label='JOREK', linestyle='--'
    )
    
    jorek_psi_spline = UnivariateSpline(jorek_rs, jorek_psi, s=0.0)
    jorek_dpsi_dr = jorek_psi_spline.derivative()(jorek_rs)
    ax_dpsi_dr.plot(
        jorek_rs, jorek_dpsi_dr/max(jorek_psi), color='red', linestyle='--'
    )
    
    ax.set_ylabel("Normalised perturbed flux")
    ax_dpsi_dr.set_ylabel("Normalised $\partial \delta\psi^{(1)}/\partial r$") 
    ax_dpsi_dr.set_xlabel("Normalised minor radial co-ordinate (a)")
    
    fig.legend()

    savefig("outer_soln_comparison")


def test_energy_calculation(params: TearingModeParameters):
    tm = solve_system(params)

    norm_energy_int = normalised_energy_integral(tm, params)

    print("Normalised energy")
    print(norm_energy_int)

    psi_rs = 1e-10

    e = energy(psi_rs, params, norm_energy_int)

    print(f"Magnetic energy: {e}")
    

def ql_tm_vs_time():
    """
    Numerically solve the quasi-linear time-dependent tearing mode problem
    and plot the perturbed flux and layer width as functions of time.
    """
    
    psi_current_prof_filename = "../../jorek_tools/postproc/exprs_averaged_s00000.csv"
    q_prof_filename = "../../jorek_tools/postproc/qprofile_s00000.dat"
    jorek_psi_filename = "../../jorek_tools/postproc/outer_region.csv"
    
    q_profile, j_profile = q_and_j_from_csv(
        psi_current_prof_filename, q_prof_filename
    )
    
    params = TearingModeParameters(
        poloidal_mode_number = 2,
        toroidal_mode_number = 1,
        lundquist_number = 4.32e6,
        initial_flux = 3e-9,
        B0=1.0,
        R0=40.0,
        q_profile = q_profile,
        j_profile = j_profile
    )

    times = np.linspace(0.0, 1e7, 100)
    
    jorek_psi_data = pd.read_csv(jorek_psi_filename)

    plot_outer_region_solution(params, jorek_psi_data)

    test_energy_calculation(params)
    
    plt.show()


if __name__=='__main__':
    ql_tm_vs_time()
