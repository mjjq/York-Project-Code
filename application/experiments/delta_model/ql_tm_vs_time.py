import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from dataclasses import asdict

import imports

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
    params = TearingModeParameters(
        poloidal_mode_number = 2,
        toroidal_mode_number = 1,
        lundquist_number = 1e8,
        axis_q = 1.0,
        profile_shaping_factor = 2.0,
        initial_flux = 1e-10
    )

    times = np.linspace(0.0, 1e8, 1000)

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

    fname = f"delta_model_(m,n,A,q0)=({params.poloidal_mode_number},{params.toroidal_mode_number},{params.initial_flux},{params.axis_q})"
    savefig(fname)
    sim_to_disk(fname, params, ql_solution)
    plt.show()


if __name__=='__main__':
    ql_tm_vs_time()
