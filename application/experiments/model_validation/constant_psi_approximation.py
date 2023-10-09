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

def constant_psi_approx():
    """
    Test the constant-psi approximation using the full (unapproximated) value
    of the discontinuity parameter for a quasi-linear tearing mode solution.
    """
    m=2
    n=1
    S=1e8
    axis_q=1.0
    r_s=rational_surface(m/n)
    s=magnetic_shear(r_s, m, n)

    fname = "../../tearing_mode_solver/output/29-08-2023_11:04_new_ql_tm_time_evo_(m,n,A,q0)=(3,2,1e-10,1.0).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

    #delta_ql_orig = mode_width(
        #ql_sol.psi_t,
        #ql_sol.dpsi_dt,
        #ql_sol.d2psi_dt2,
        #r_s,
        #m,
        #n,
        #s,
        #S
    #)
    w = island_width(
        ql_sol.psi_t,
        r_s,
        m,
        n,
        s
    )

    #dps = delta_prime_non_linear(ql_sol, w)

    d_delta_primes = w * ql_sol.delta_primes

    fig_dp, ax_dp = plt.subplots(1)
    ax_dp.plot(ql_sol.times, ql_sol.delta_primes)

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.set_xscale('log')

    ax.plot(ql_sol.times, d_delta_primes, color='black')

    ax.set_xlabel(r'Normalised time ($1/\bar{\omega}_A$)')
    ax.set_ylabel(r"$w(t) \Delta'[w(t)]$")

    fig.tight_layout()

    orig_fname, ext = os.path.splitext(os.path.basename(fname))
    savefig(f"{orig_fname}_const_psi_approx")

if __name__=='__main__':
    constant_psi_approx()
