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
from plot_delql_terms import plot_delql_terms

def compare_delql_terms():
    """
    Plot the full quasi-linear layer width as a function of (X, t) on a heatmap.
    """
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "../../tearing_mode_solver/output/28-08-2023_19:29_new_ql_tm_time_evo_(m,n,A,q0)=(2,1,1e-10,1.0).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

    ts_to_plot = [100.0, 1e3, 1e5, 5e7]
    for t in ts_to_plot:
        plot_delql_terms(
            ql_sol,
            m,n,
            S,
            r_s,
            (-100, 100),
            t
        )

if __name__=='__main__':
    compare_delql_terms()
