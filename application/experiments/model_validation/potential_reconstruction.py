from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from scipy.integrate import quad, simpson
from tqdm import tqdm, trange
from typing import Tuple
import os
import numpy as np
import pandas as pd

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
from tearing_mode_solver.phi_reconstruction import (
    potential, check_solution_is_valid
)

def potential_from_data():
    """
    Load pre-computed quasi-linear solution data, re-construct the electric
    potential, then plot the function as a heatmap.
    """
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "../../tearing_mode_solver/output/29-08-2023_10:53_new_ql_tm_time_evo_(m,n,A,q0)=(2,1,1e-10,1.0).csv"
    df = pd.read_csv(fname)
    df = df.iloc[:100000:1000,:]
    ql_sol = classFromArgs(TimeDependentSolution, df)
    ql_sol.w_t = mode_width(
        ql_sol.psi_t,
        ql_sol.dpsi_dt,
        ql_sol.d2psi_dt2,
        r_s,
        m,
        n,
        s,
        S
    )

    xs = np.linspace(-0.5, 0.5, 1000)
    #print(xs)
    phi = potential(ql_sol, n, s, xs)

    fig, ax = plt.subplots(1, figsize=(4,3))
    times = ql_sol.times
    im = ax.imshow(
        phi,
        extent=[min(times), max(times), min(xs), max(xs)]
    )

    fig.colorbar(im)
    ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))

    ax.set_xlabel(r"Normalised time ($1/\bar{\omega}_A$)")
    ax.set_ylabel(r"x ($r_s$)")

    fig.tight_layout()

    orig_fname, ext = os.path.splitext(os.path.basename(fname))
    savefig(f"potential_rec_{orig_fname}")

    check_solution_is_valid(
        phi,
        xs,
        times,
        ql_sol.dpsi_dt,
        ql_sol.w_t,
        n,
        s
    )

    orig_fname, ext = os.path.splitext(os.path.basename(fname))
    savefig(f"pde_validation_{orig_fname}")

if __name__=='__main__':
    potential_from_data()
