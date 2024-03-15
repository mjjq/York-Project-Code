import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from dataclasses import asdict
from os.path import join
from scipy.interpolate import UnivariateSpline

import imports

from tearing_mode_solver.profiles import j
from jorek_tools.jorek_dat_to_array import q_and_j_from_input_files
from tearing_mode_solver.delta_model_solver import solve_time_dependent_system
from tearing_mode_solver.helpers import (
    savefig, 
    savecsv, 
    TearingModeParameters,
    sim_to_disk
)

def ql_tm_vs_time():
    """
    Numerically solve the quasi-linear time-dependent tearing mode problem
    and plot the perturbed flux and layer width as functions of time.
    """
    
    psi_current_prof_filename = "../../jorek_tools/postproc/exprs_averaged_s00000.dat"
    q_prof_filename = "../../jorek_tools/postproc/qprofile_s00000.dat"
    q_profile, j_profile = q_and_j_from_input_files(
        psi_current_prof_filename, q_prof_filename
    )
    
    
    #rq, q = zip(*q_profile)
    #q = np.array(q)/q[0]
    #rj, js = zip(*j_profile)  
    #fig, ax = plt.subplots(3)
    #ax[0].plot(rq, q)
    #ax[1].plot(rj, js)
    #ax[2].plot(rj, dj_dr_vals)
    
    
    params = TearingModeParameters(
        poloidal_mode_number = 2,
        toroidal_mode_number = 1,
        lundquist_number = 1.147e10,
        initial_flux = 1e-10,
        B0=1.0,
        R0=40.0,
        q_profile = q_profile,
        j_profile = j_profile
    )

    times = np.linspace(0.0, 1e8, 10000)

    ql_solution = solve_time_dependent_system(
        params, times
    )
    times = ql_solution.times
    psi_t = ql_solution.psi_t
    dpsi_t = ql_solution.dpsi_dt
    w_t = ql_solution.w_t

    print(times)
    print(ql_solution.psi_t)
    #plt.plot(w_t)
    #plt.show()

    #print("lgr: ", lin_growth_rate)
    #print("Threshold: ", ql_threshold)
    #print(psi_t)

    fig, ax = plt.subplots(1, figsize=(4,3.5))
    ax2 = ax.twinx()


    ax.plot(times, psi_t, label='Flux', color='black')

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \psi^{(1)}$)")

    ax2.plot(times, w_t, label='Normalised island width', color='red')
    ax2.set_ylabel(r"Normalised electrostatic modal width ($\hat{\delta}_{ql}$)")
    ax2.yaxis.label.set_color('red')

    ax.legend(prop={'size': 7}, loc='lower right')

    # ax.set_yscale('log')
    # ax2.set_yscale('log')
    ax.set_xscale('log')
    ax2.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_yscale('log')

    fig.tight_layout()
    #plt.show()

    fname = f"jorek_model_(m,n)=({params.poloidal_mode_number},{params.toroidal_mode_number})"
    savefig(fname)
    sim_to_disk(fname, params, ql_solution)
    plt.show()


if __name__=='__main__':
    ql_tm_vs_time()
