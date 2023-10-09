import numpy as np
from dataclasses import dataclass, fields
import pandas as pd
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from scipy.integrate import quad, simpson
from tqdm import tqdm, trange
from typing import Tuple
import os

from tearing_mode_solver.y_sol import Y
from tearing_mode_solver.delta_model_solver import nu, mode_width
from tearing_mode_solver.helpers import (
    savefig, classFromArgs, TimeDependentSolution
)
from tearing_mode_solver.outer_region_solver import (
    magnetic_shear, rational_surface,
    island_width, delta_prime_non_linear
)

def del_ql_full(sol: TimeDependentSolution,
                poloidal_mode: int,
                toroidal_mode: int,
                lundquist_number: float,
                mag_shear: float,
                r_s: float,
                x_range: Tuple[float, float],
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

    xmin, xmax = x_range
    xs = np.arange(xmin, xmax, dx)
    ys = Y(xs)
    ys_func = UnivariateSpline(xs, ys, s=0)
    d2ydx2 = ys_func.derivative().derivative()
    d3ydx3 = d2ydx2.derivative()

    del_dot_term = dw_dt_func(times)/w_t_func(times)
    psi_dot_term = sol.d2psi_dt2/sol.dpsi_dt
    nu_value = nu(sol.psi_t, poloidal_mode, lundquist_number, r_s)

    pre_factor = 1.0/( lundquist_number * (toroidal_mode*mag_shear)**2)

    deltas = []

    tqdm_range = trange(len(xs), leave=True)
    for i in tqdm_range:
        x = xs[i]
        M = x*d3ydx3(x)/d2ydx2(x)
        del_dot = - del_dot_term * (3.0+M)
        psi_dot = psi_dot_term

        delta_pow_4 = pre_factor*(nu_value + psi_dot + del_dot)

        delta_pow_4[delta_pow_4<0.0] = 0.0

        delta_at_x = delta_pow_4**(1/4)

        deltas.append(delta_at_x)

    deltas = np.array(deltas)

    return deltas, times, xs

def simple_integration():
    """
    Perform the approximate spatial integral for the inner layer solution to
    verify that we get the correct result. We should get ~2.12
    """
    xs = np.linspace(-10.0, 10.0, 100)
    ys = Y(xs)
    
    int_result = simpson(
        (1.0+xs*ys), x=xs
    )
    
    print(int_result)
    

def delta_prime_full(delta_qls: np.ndarray,
                     xs: np.array,
                     times: np.array,
                     psi_t: np.array,
                     dpsi_dt: np.array,
                     w_t: np.array,
                     r_s: float,
                     lundquist_number: float):
    """
    Calculate the full (unapproximated) discontinuity parameter of the
    quasi-linear inner layer solution. Integrates over the spatial component
    so that Delta' is a function of time only.

    Parameters:
        deltas: np.ndarray
            2D array containing delta values as a function of X and t. First
            dimension corresponds to time, second dimension corresponds to
            space.
        xs: np.array
            Array of X values associated with the second dimension of deltas
        times: np.array
            Array of times associated with the first dimension of deltas
        psi_t: np.array
            Perturbed flux at the resonant surface as a function of time
        dpsi_dt: np.array
            First time derivative in perturbed flux at resonant surface as a
            function of time.
        w_t: np.array
            Quasi-linear layer width as a function of time.
        r_s: float
            Location of the resonant surface normalised to the minor radius of
            the plasma.
        lundquist_number: float
            The Lundquist number.

    Returns:
        delta_primes: np.array
            Discontinuity parameter as a function of time.
    """
    
    ys = Y(xs)
    
    
    delta_primes = []
    tqdm_range = trange(len(times), leave=True)
    for i in tqdm_range:
        t= times[i]
        delta_ql_x = delta_qls[:,i]
        delta_orig = w_t[i]
        
        int_result = simpson(
            (1.0+(delta_orig/delta_ql_x)*xs*ys), x=xs
        )

        psi = psi_t[i]
        dpsi = dpsi_dt[i]
        
        delta_primes.append(lundquist_number*dpsi*delta_orig*int_result/(psi*r_s))
    
    return np.array(delta_primes)




if __name__=='__main__':
    #constant_psi_approx()
    #convergence_of_delta_prime()
    constant_psi_approx()
    #compare_delql_terms()

    plt.show()
