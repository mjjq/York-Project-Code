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


def plot_delql_terms(sol: TimeDependentSolution,
                poloidal_mode: int,
                toroidal_mode: int,
                lundquist_number: float,
                r_s: float,
                mag_shear: float,
                x_range: Tuple[float, float],
                plot_t: float,
                dx: float = 0.01):
    """
    Calculate the full (unapproximated) quasi-linear layer width as a function
    of X and t.

    Parameters:
        sol: TimeDependentSolution
            The time-dependent tearing mode solution
        poloidal_mode: int
            Poloidal mode number of the tearing mode
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        lundquist_number: float
            The Lundquist number
        mag_shear: float
            Magnetic shear at the resonant surface
        r_s: float
            Location of the resonant surface normalised to the minor radius of
            the plasma.
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

    times = sol.times
    w_t_func = UnivariateSpline(times, sol.w_t, s=0)
    dw_dt_func = w_t_func.derivative()

    psi_t_func = UnivariateSpline(times, sol.psi_t, s=0)
    dpsi_dt_func = psi_t_func.derivative()
    d2psi_dt2_func = dpsi_dt_func.derivative()

    dpsi_dt_vals = dpsi_dt_func(times)
    d2psi_dt2_vals = d2psi_dt2_func(times)

    delta_t = mode_width(
        sol.psi_t,
        dpsi_dt_vals,
        d2psi_dt2_vals,
        r_s,
        poloidal_mode,
        toroidal_mode,
        mag_shear,
        lundquist_number
    )
    delta_t_func = UnivariateSpline(times, delta_t, s=0)
    ddelta_dt_func = delta_t_func.derivative()

    #fig_delta, ax_delta = plt.subplots(1, figsize=(4,3))
    #ax_delta.plot(times, ddelta_dt_func(times)/delta_t_func(times))
    #ax_delta.set_xlabel(r'Time ($1/\bar{\omega}_A$)')
    #ax_delta.set_ylabel(r'Electrostatic mode growth rate ($\bar{\omega}_A$)')
    #fig_delta.tight_layout()
    #ax_delta.set_xlim(left=0.0, right=20000.0)

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
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "../../tearing_mode_solver/output/28-08-2023_19:29_new_ql_tm_time_evo_(m,n,A,q0)=(2,1,1e-10,1.0).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

    t = 5e7

    plot_delql_terms(
        ql_sol,
        m,n,
        S,
        r_s,
        s,
        (-10, 10),
        t
    )

    plt.show()
