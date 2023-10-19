import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from dataclasses import asdict

import imports

from tearing_mode_solver.delta_model_solver import (
    solve_time_dependent_system,
    mode_width_precalc
)
from tearing_mode_solver.helpers import (
    savefig, 
    savecsv, 
    TearingModeParameters,
    sim_to_disk,
    load_sim_from_disk
)

def ql_tm_vs_time():
    """
    Numerically solve the quasi-linear time-dependent tearing mode problem
    and plot the perturbed flux and layer width as functions of time.
    """

    fname = f"/home/marcus/Nextcloud/Documents/Fusion Energy MSc/Courses/Project/Code/application/experiments/delta_model/output/19-10-2023_16:27_delta_model_(m,n,A,q0)=(2,1,1e-10,1.0).zip"

    params, ql_solution = load_sim_from_disk(fname)

    times = ql_solution.times
    psi_t = ql_solution.psi_t
    w_t = ql_solution.w_t


    fig, ax = plt.subplots(1, figsize=(4,3.5))
    ax2 = ax.twinx()

    t_filter = ((times > 5e4) & (times < 2e5))
    times_filtered = times[t_filter]
    psi_t_filt = psi_t[t_filter]



    ax.plot(times_filtered, psi_t_filt, label='Flux', color='black')

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \psi^{(1)}$)")

    mode_width = mode_width_precalc(
        params,
        ql_solution
    )
    mode_width_filtered = mode_width[t_filter]


    ax2.plot(
        times_filtered, mode_width_filtered,
        label='Normalised island width', color='red'
    )
    ax2.set_ylabel(r"Normalised electrostatic modal width ($\hat{\delta}_{ql}$)")
    ax2.yaxis.label.set_color('red')

    ax.legend(prop={'size': 7}, loc='lower right')

    # ax.set_yscale('log')
    # ax2.set_yscale('log')
    ax.set_yscale('log')
    ax2.set_yscale('log')

    #ax.set_xlim(left=1e5, right=2e5)
    #ax2.set_xlim(left=1e5, right=2e5)
    #ax.set_ylim(left=
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig.tight_layout()
    #plt.show()

    fname = f"delta_model_(m,n,A,q0)=({params.poloidal_mode_number},{params.toroidal_mode_number},{params.initial_flux},{params.axis_q})"
    savefig(fname)
    #sim_to_disk(fname, params, ql_solution)
    plt.show()


if __name__=='__main__':
    ql_tm_vs_time()
