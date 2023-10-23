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


def plot_delql_terms(sol: TimeDependentSolution,
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
    w_t_func = UnivariateSpline(times, sol.w_t, s=0)
    dw_dt_func = w_t_func.derivative()

    psi_t_func = UnivariateSpline(times, sol.psi_t, s=0)
    dpsi_dt_func = psi_t_func.derivative()
    d2psi_dt2_func = dpsi_dt_func.derivative()

    dpsi_dt_vals = dpsi_dt_func(times)
    d2psi_dt2_vals = d2psi_dt2_func(times)

    delta_t = mode_width_precalc(
        params,
        sol
    )
    delta_t_func = UnivariateSpline(times, delta_t, s=0)
    ddelta_dt_func = delta_t_func.derivative()

    #fig_delta, ax_delta = plt.subplots(1, figsize=(4,3))
    #ax_delta.plot(times, ddelta_dt_func(times)/delta_t_func(times))
    #ax_delta.set_xlabel(r'Time ($1/\bar{\omega}_A$)')
    #ax_delta.set_ylabel(r'Electrostatic mode growth rate ($\bar{\omega}_A$)')
    #fig_delta.tight_layout()
    #ax_delta.set_xlim(left=0.0, right=20000.0)

    #ax_just_delta = ax_delta.twinx()
    #ax_just_delta.plot(times, delta_t_func(times))

    #plt.plot(times, delta_t_func(times))
    #plt.plot(times, ddelta_dt_func(times)/delta_t_func(times))

    xmin, xmax = x_range
    xs = np.arange(xmin, xmax, dx)
    ys = Y(xs)
    ys_func = UnivariateSpline(xs, ys, s=0)
    d2ydx2 = ys_func.derivative().derivative()
    d3ydx3 = d2ydx2.derivative()


    del_dot_term = ((xs*d3ydx3(xs) + 3.0*d2ydx2(xs))*
        ddelta_dt_func(plot_t)/delta_t_func(plot_t))
    psi_dot_term = d2ydx2(xs)*d2psi_dt2_func(plot_t)/dpsi_dt_func(plot_t)
    nu_value = (
        d2ydx2(xs)*nu(psi_t_func(plot_t), poloidal_mode, lundquist_number, r_s)
    )

    #fig_derivs, ax_derivs = plt.subplots(1)
    #ax_derivs.plot(xs, xs*d3ydx3(xs)+3.0*d2ydx2(xs))

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(
        xs, nu_value+psi_dot_term,
        label=r"$\nu Y'' + Y'' \delta\ddot{\psi}/\delta\dot{\psi}$"
    )
    #ax.plot(
    #    times, psi_dot_term,
    #    label=r"$|Y'' \delta\ddot{\psi}/\delta\dot{\psi}|$"
    #)
    ax.plot(
        xs, del_dot_term,
        label=r"$[XY''' + 3Y''] \dot{\delta}/\delta$"
    )

    ax.set_xlabel(r"X")
    ax.set_ylabel(r"Contribution to $Y'' \delta^4_{ql}(X, t)$")

    ax.set_title(r"$\bar{\omega}_A t$"f"={plot_t:.1f}")

    ax.legend(prop={'size':8})

    #ax.set_xscale('log')
    #ax.set_yscale('log')

    fig.tight_layout()

    savefig(f"delql_contributions_t={plot_t:.2f}")
    #plt.show()

if __name__=='__main__':
    fname = "/home/marcus/Nextcloud/Documents/Fusion Energy MSc/Courses/Project/Code/application/experiments/delta_model/output/19-10-2023_16:27_delta_model_(m,n,A,q0)=(2,1,1e-10,1.0).zip"
    params, sol = load_sim_from_disk(fname)
    #ql_sol = classFromArgs(TimeDependentSolution, df)

    ts = np.linspace(0.5e5, 1.0e5, 5)
    #t = 1.4e5

    for t in ts:
        plot_delql_terms(
            sol,
            params,
            (-10, 10),
            t
        )

    plt.show()
