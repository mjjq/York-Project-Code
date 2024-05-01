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

def ql_tm_vs_time():
    """
    Plot various numerically solved variables from a tearing mode solution and
    island width as a function of time from .csv data.
    """
    model_data_filename = "./output/30-04-2024_13:26_jorek_model_(m,n)=(2,1).zip"
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

    model_filt = times>0.0
    times = times[model_filt]
    model_growth_rate = model_growth_rate[model_filt]

    jorek_filt = jorek_times>min_time
    jorek_times = jorek_times[jorek_filt]
    jorek_growth_rate = jorek_growth_rate[jorek_filt]

    fig, ax = plt.subplots(1)

    ax.plot(times, model_growth_rate, label='Model')
    ax.plot(jorek_times, jorek_growth_rate, label='JOREK')

    ax.legend()

    plt.show()





if __name__=='__main__':
    ql_tm_vs_time()
