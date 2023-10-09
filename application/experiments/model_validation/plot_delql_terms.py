from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from scipy.integrate import quad, simpson
from tqdm import tqdm, trange
from typing import Tuple
import os
import numpy as np

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

    xmin, xmax = x_range
    xs = np.arange(xmin, xmax, dx)
    ys = Y(xs)
    ys_func = UnivariateSpline(xs, ys, s=0)
    d2ydx2 = ys_func.derivative().derivative()
    d3ydx3 = d2ydx2.derivative()


    del_dot_term = ((xs*d3ydx3(xs) + 3.0*d2ydx2(xs))*
        dw_dt_func(plot_t)/w_t_func(plot_t))
    psi_dot_term = d2ydx2(xs)*d2psi_dt2_func(plot_t)/dpsi_dt_func(plot_t)
    nu_value = (
        d2ydx2(xs)*nu(psi_t_func(plot_t), poloidal_mode, lundquist_number, r_s)
    )

    #fig_derivs, ax_derivs = plt.subplots(1)
    #ax_derivs.plot(xs, xs*d3ydx3(xs)+3.0*d2ydx2(xs))

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(
        xs, abs(nu_value+psi_dot_term),
        label=r"$|\nu Y'' + Y'' \delta\ddot{\psi}/\delta\dot{\psi}|$"
    )
    #ax.plot(
    #    times, psi_dot_term,
    #    label=r"$|Y'' \delta\ddot{\psi}/\delta\dot{\psi}|$"
    #)
    ax.plot(
        xs, abs(del_dot_term),
        label=r"$|[XY''' + 3Y''] \dot{\delta}/\delta|$"
    )

    ax.set_xlabel(r"X")
    ax.set_ylabel(r"Contribution to $Y'' \delta^4_{ql}(X, t)$")

    ax.set_title(r"$\bar{\omega}_A t$"f"={plot_t:.1f}")

    ax.legend(prop={'size':8})

    #ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()

    savefig(f"delql_contributions_t={plot_t:.2f}")
