import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

import imports
from tearing_mode_solver.outer_region_solver import (
    rational_surface, magnetic_shear
)
from tearing_mode_solver.delta_model_solver import nu, mode_width
from tearing_mode_solver.helpers import (
    classFromArgs, TimeDependentSolution, savefig
)
from tearing_mode_solver.outer_region_solver import island_width


def difference_in_flux_models():
    """
    Load numerical solutions to a tearing mode calculated using the gamma
    and delta models, calculate the absolute difference in the perturbed flux
    as a function of time, then plot the difference.
    """
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname_new = "./output/28-08-2023_19:29_new_ql_tm_time_evo_(m,n,A,q0)=(2,1,1e-10,1.0).csv"
    df_new = pd.read_csv(fname_new)
    ql_sol_new = classFromArgs(TimeDependentSolution, df_new)

    fname_approx = "./output/28-08-2023_19:36_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df_approx = pd.read_csv(fname_approx)
    ql_sol_approx = classFromArgs(TimeDependentSolution, df_approx)

    psi_t_new_func = UnivariateSpline(
        ql_sol_new.times, ql_sol_new.psi_t, s=0
    )
    psi_t_approx_func = UnivariateSpline(
        ql_sol_approx.times, ql_sol_approx.psi_t, s=0
    )

    times = ql_sol_new.times
    pc_delta = 100.0*(psi_t_approx_func(times)/psi_t_new_func(times) - 1.0)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(times, pc_delta, color='black')
    ax.set_xscale('log')
    ax.set_xlabel(r'Normalised time ($1/\bar{\omega}_A$)')
    ax.set_ylabel(r'Flux change from gamma to delta model (%)')
    fig.tight_layout()
    savefig("flux_delta")

    fig2, ax2 = plt.subplots(1)
    ax2.plot(times, psi_t_new_func(times))
    ax2.plot(times, psi_t_approx_func(times))
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.show()



if __name__=='__main__':
    difference_in_flux_models()
