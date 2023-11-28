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
from tearing_mode_solver.delta_model_solver import nu, mode_width_precalc
from tearing_mode_solver.helpers import (
    savefig, classFromArgs, TimeDependentSolution, TearingModeParameters,
    load_sim_from_disk
)
from tearing_mode_solver.outer_region_solver import (
    magnetic_shear, rational_surface,
    island_width, delta_prime_non_linear
)
from tearing_mode_solver.profiles import rational_surface_of_mode


def verify_time_ind_q(sol: TimeDependentSolution,
                params: TearingModeParameters,
                x_range: Tuple[float, float],
                plot_t: float,
                dx: float = 0.01):
    """
    Calculate the full (unapproximated) quasi-linear layer width as a function
    of X and t.

    Parameters:
        sol: TimeDependentSolution
            The time-dependent tearing mode solution
        params: TearingModeParameters
            The tearing mode parameters
        x_range: Tuple[float, float]
            Minimum and maximum bounds defining the range of X values to use in
            the calculation of delta(X,t)
        dx: float
            Distance between adjacent X-values in x_range

    Returns:
        deltas: np.ndarray
            2D array containing delta values as a function of X and t. First
            dimension corresponds to time, second dimension corresponds to
            space.
        times: np.array
            Array of times associated with the first dimension of deltas
        xs: np.array
            Array of X values associated with the second dimension of deltas

    """
    poloidal_mode = params.poloidal_mode_number
    toroidal_mode = params.toroidal_mode_number
    shaping_exponent = params.profile_shaping_factor
    lundquist_number = params.lundquist_number

    r_s = rational_surface_of_mode(
        poloidal_mode,
        toroidal_mode,
        params.axis_q,
        shaping_exponent
    )
    mag_shear = magnetic_shear(
        r_s,
        poloidal_mode,
        toroidal_mode,
        shaping_exponent
    )


    times = sol.times

    psi_t_func = UnivariateSpline(times, sol.psi_t, s=0)
    dpsi_dt_func = psi_t_func.derivative()

    psi_t_vals = psi_t_func(times)
    dpsi_dt_vals = dpsi_dt_func(times)

    delta_t = mode_width_precalc(
        params,
        sol
    )
    delta_t_func = UnivariateSpline(times, delta_t, s=0)

    xmin, xmax = x_range
    xs = np.arange(xmin, xmax, dx)
    ys = Y(xs)
    ys_func = UnivariateSpline(xs, ys, s=0)

    pre_factor = poloidal_mode**2/(
        2.0*lundquist_number*r_s**4 * toroidal_mode**2 * mag_shear**2
    )
    spatial_term = ys/xs
    temporal_term = dpsi_dt_func(times)*psi_t_func(times)\
        /(delta_t_func(times))**2

    outer_product = np.outer(spatial_term, temporal_term)
    full_term = np.abs(pre_factor * outer_product)

    fig, ax = plt.subplots(1)
    im = ax.imshow(
        full_term,
        extent=[min(times), max(times), min(xs), max(xs)]
    )

    fig.colorbar(im)
    ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))

    ax.set_xlabel(r"Normalised time ($1/\bar{\omega}_A$)")
    ax.set_ylabel(r"X ($r_s$)")

    fig.tight_layout()

    #fig, ax = plt.subplots(1, figsize=(4,3))

    #ax.plot(
    #    xs, full_term
    #)

    print(np.max(full_term))

    #ax.set_xlabel(r"X")
    #ax.set_ylabel(r"Temporal component of $\epsilon(x,t)/\mu x$")

    #ax.set_title(r"$\bar{\omega}_A t$"f"={plot_t:.1f}")

    #ax.legend(prop={'size':8})

    ##ax.set_xscale('log')
    ##ax.set_yscale('log')

    #fig.tight_layout()

    savefig(f"q_time_verification_(m,n)=({poloidal_mode},{toroidal_mode})")
    #plt.show()

if __name__=='__main__':
    fname = "/home/marcus/Nextcloud/Documents/Fusion Energy MSc/Courses/Project/Code/application/experiments/delta_model/output/19-10-2023_16:27_delta_model_(m,n,A,q0)=(2,1,1e-10,1.0).zip"
    params, sol = load_sim_from_disk(fname)
    #ql_sol = classFromArgs(TimeDependentSolution, df)

    #ts = np.linspace(2.5e5, 2.5e6, 4)
    ts = [sol.times[-1]]

    for t in ts:
        verify_time_ind_q(
            sol,
            params,
            (-10, 10),
            t
        )

    plt.show()
