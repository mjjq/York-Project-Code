import numpy as np
from dataclasses import dataclass, fields
import pandas as pd
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from scipy.integrate import quad, simpson
from tqdm import tqdm, trange
from typing import Tuple
import os

import imports
from tearing_mode_solver.y_sol import Y
from tearing_mode_solver.delta_model_solver import nu, mode_width
from tearing_mode_solver.helpers import (
    savefig, classFromArgs, TimeDependentSolution
)
from tearing_mode_solver.outer_region_solver import (
    magnetic_shear, rational_surface,
    island_width, delta_prime_non_linear
)
from tearing_mode_solver.unapprox_layer_width import(
    simple_integration,
    del_ql_full,
    delta_prime_full
)

def convergence_of_growth_rate():
    """
    Demonstrate convergence of the unapproximated growth rate to the
    approximated value over a numerical solution to the quasi-linear equations.
    """
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "../../tearing_mode_solver/output/18-08-2023_16:41_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

    delta_ql_orig = island_width(
        ql_sol.psi_t,
        r_s,
        m,
        n,
        s
    )

    simple_integration()

    delqls, times, xs = del_ql_full(ql_sol, m, n, S, s, r_s, (-1.0, 1.0))
    #delqls = delqls[:,::1000]

    delta_primes = delta_prime_full(
        delqls,
        xs,
        times,
        ql_sol.psi_t,
        ql_sol.dpsi_dt,
        ql_sol.w_t,
        r_s,
        S
    )

    fig_dp, ax_dp = plt.subplots(1, figsize=(4,4))
    ax_dp.plot(times, delta_primes, label=r"Exact $\Delta'$")
    ax_dp.plot(times, ql_sol.delta_primes, label=r"Approximate $\Delta'$")

    # fig, ax = plt.subplots(1, figsize=(4,4))
    # plt.imshow(
    #     delqls,
    #     extent=[min(times), max(times), min(xs), max(xs)]
    # )

    # ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))

    fig2, ax2 = plt.subplots(1, figsize=(4,3))

    ax2.plot(times, delta_ql_orig, label=f'Recreated approximate soln')
    ax2.plot(times, ql_sol.w_t, label=f'Approximate solution')
    ax2.plot(times, delqls[-1,:], label=f'x={xs[-1]:.2f}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    for i in range(len(xs)//2, len(xs), 200):
        lineout = delqls[i,:]
        ax2.plot(times, lineout, label=f'x={xs[i]:.2f}')


    ax2.legend()

    fig_psi, ax_psi = plt.subplots(1)
    ax_psi.plot(times, ql_sol.psi_t)
    ax_w = ax_psi.twinx()

    ax_psi.set_xscale('log')
    ax_psi.set_yscale('log')

    ax_w.plot(times, ql_sol.w_t)
    ax_w.set_yscale('log')

    plt.show()

if __name__=='__main__':
    convergence_of_growth_rate()
