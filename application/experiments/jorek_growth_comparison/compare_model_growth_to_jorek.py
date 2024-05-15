import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import os
import numpy as np

import imports
from tearing_mode_solver.outer_region_solver import (
    rational_surface, magnetic_shear
)
from tearing_mode_solver.delta_model_solver import nu, mode_width
from tearing_mode_solver.helpers import (
    classFromArgs, TimeDependentSolution, savefig, load_sim_from_disk
)
from tearing_mode_solver.outer_region_solver import island_width
from jorek_tools.calc_jorek_growth import growth_rate, _name_time

def check_model_t_dependence():
    model_data_filename = "./output/14-05-2024_16:59_jorek_model_(m,n)=(2,1).zip"

    params, sol = load_sim_from_disk(model_data_filename)

    times = sol.times
    psi_t = sol.psi_t
    dpsi_t = sol.dpsi_dt
    w_t = sol.w_t
    d2psi_dt2 = sol.d2psi_dt2
    delta_primes = sol.delta_primes

    print(delta_primes)

    fig, ax = plt.subplots(1)

    ax.plot(times, psi_t)
    ax.set_xscale('log')

    plt.show()

def ql_tm_vs_time():
    """
    Plot various numerically solved variables from a tearing mode solution and
    island width as a function of time from .csv data.
    """
    model_data_filename = "./output/14-05-2024_18:18_jorek_model_(m,n)=(2,1).zip"
    jorek_data_filename = "../../jorek_tools/postproc/magnetic_energies.csv"

    params, sol = load_sim_from_disk(model_data_filename)

    times = sol.times
    psi_t = sol.psi_t
    dpsi_t = sol.dpsi_dt
    w_t = sol.w_t
    d2psi_dt2 = sol.d2psi_dt2
    delta_primes = sol.delta_primes

    model_growth_rate = dpsi_t/psi_t

    jorek_data = pd.read_csv(jorek_data_filename)
    jorek_growth_rate = growth_rate(jorek_data)
    jorek_times = jorek_data[_name_time]

    min_time = 1e4

    model_filt = ((times<1e6) & (times> 1e4))
    times = times[model_filt]
    model_growth_rate = model_growth_rate[model_filt]

    jorek_filt = jorek_times>min_time
    jorek_times = jorek_times[jorek_filt]
    jorek_growth_rate = jorek_growth_rate[jorek_filt]

    fig, ax = plt.subplots(1)

    ax.plot(times, model_growth_rate, label='Model', color='black')
    ax.plot(jorek_times, jorek_growth_rate, label='JOREK', color='red')

    log_times = np.logspace(4, np.log(max(times)), 100)
    #ax.plot(log_times, 1.0/log_times, label='2/t dependence')

    ax.legend()

    ax.set_xscale('log')
    #ax.set_yscale('log')

    ax.set_xlabel(f"Time ($1/\omega_A$)")
    ax.set_ylabel(f"Growth rate ($\omega_A$)")

    fig.tight_layout()

    savefig("growth_comparison")

    plt.show()





if __name__=='__main__':
    ql_tm_vs_time()
    #check_model_t_dependence()
