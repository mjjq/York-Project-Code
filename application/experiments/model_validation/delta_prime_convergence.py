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

def convergence_of_delta_prime():
    """
    Demonstrate convergence of the unapproximated discontinuity parameter to the
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

    x_lims = [1.0, 1.5, 5.0]


    fig_dp, ax_dp = plt.subplots(1, figsize=(4,4))
    ax_dp.plot(
        ql_sol.times,
        ql_sol.delta_primes,
        label=r"Approximate $a\Delta'$",
        color='black'
    )
    ax_dp.set_xscale('log')

    for xlim in x_lims:
        delqls, times, xs = del_ql_full(
            ql_sol, m, n, S, s, r_s,
            (-xlim, xlim),
            0.1
        )
        #delqls = delqls[:,::1000]

        delta_primes = delta_prime_full(
            delqls,
            xs,
            times,
            ql_sol.psi_t,
            ql_sol.dpsi_dt,
            delta_ql_orig,
            r_s,
            S
        )

        times_f = times[times>1e3]
        delta_primes_f = delta_primes[-len(times_f):]
        print(times_f.shape)
        print(delta_primes_f.shape)
        ax_dp.plot(
            times_f, delta_primes_f, label=r"Exact $a\Delta'$"+f"(X={xlim})"
        )

    ax_dp.legend()
    ax_dp.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    ax_dp.set_ylabel(r"$a\Delta'$")
    ax_dp.set_xlim(left=1e3)
    #ax_dp.set_ylim(bottom=1.0)
    fig_dp.tight_layout()

    orig_fname, ext = os.path.splitext(os.path.basename(fname))
    savefig(f"{orig_fname}_delta_prime_convergence")

if __name__=='__main__':
    convergence_of_delta_prime()
