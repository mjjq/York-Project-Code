#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:06:24 2023

@author: marcus
"""
import numpy as np
from typing import Tuple
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from scipy.stats import sem
from scipy.optimize import curve_fit
from typing import Tuple


from linear_solver import (
    TearingModeSolution, solve_system, magnetic_shear,
    scale_tm_solution, delta_prime, q
)
from non_linear_solver import delta_prime_non_linear, island_width
from helpers import savefig, TimeDependentSolution, dataclass_to_disk


def flux_time_derivative(psi: float,
                         time: float,
                         tm: TearingModeSolution,
                         poloidal_mode: int,
                         toroidal_mode: int,
                         lundquist_number: float,
                         mag_shear: float,
                         epsilon: float = 1e-5):
    """
    Calculate first order time derivative of the perturbed flux
    in the inner region of the tearing mode using the strongly non-linear
    time evolution equation.

    Parameters:
        psi: float
            Tuple containing the perturbed flux at the current time
        time: float
            The current time of the simulation.
        tm: TearingModeSolution
            The outer solution of the current tearing mode
        poloidal_mode: int
            Poloidal mode number of the tearing mode
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        lundquist_number: float
            The Lundquist number
        mag_shear: float
            Magnetic shear at the resonant surface. See magnetic_shear() in
            linear_solver.py
    """
    
    m = poloidal_mode
    n = toroidal_mode
      
    s = mag_shear
    w = island_width(psi, tm.r_s, m, n, s)
    delta_prime = delta_prime_non_linear(tm, w)
    sqrt_factor = (tm.r_s**3)*s*psi
    
    
    if sqrt_factor >= 0.0:
        dpsi_dt = (0.5*1.12*((n*s/m)**0.5)*(psi**0.5)*tm.r_s**2 *
            (delta_prime/lundquist_number)
        )

    else:
        dpsi_dt = 0.0
        
    
    return dpsi_dt


def solve_time_dependent_system(poloidal_mode: int, 
                                toroidal_mode: int, 
                                lundquist_number: float,
                                axis_q: float,
                                initial_scale_factor: float = 1.0,
                                t_range: np.array = np.linspace(0.0, 1e5, 10))\
    -> Tuple[TimeDependentSolution, TearingModeSolution]:
    """
    Numerically integrate the quasi-linear flux time derivative of a tearing
    mode.

    Parameters:
        poloidal_mode: int
                Poloidal mode number of the tearing mode
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        lundquist_number: float
            The Lundquist number
        axis_q: float
            The on-axis equilibrium safety factor
        initial_scale_factor: float
            The value of the perturbed flux at the resonant surface at t=0
        t_range: np.array
            Array of time values to record. Each element will have an associated
            perturbed flux, derivative etc calculated for that time.
    """
    
    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)

    psi_t0 = initial_scale_factor
    
    s = magnetic_shear(tm.r_s, poloidal_mode, toroidal_mode)
    
    psi_t = odeint(
        flux_time_derivative,
        psi_t0,
        t_range,
        args = (tm, poloidal_mode, toroidal_mode, lundquist_number, s)
    )
    
    # We get weird numerical bugs sometimes returning large or nan values.
    # Set these to zero.
    psi_t[np.abs(psi_t) > 1e10] = 0.0
    psi_t[np.argwhere(np.isnan(psi_t))] = 0.0

    w_t = np.squeeze(
        island_width(psi_t, tm.r_s, poloidal_mode, toroidal_mode, s)
    )

    dps = [delta_prime_non_linear(tm, w) for w in w_t]
    

    return TimeDependentSolution(
        t_range,
        np.squeeze(psi_t),
        np.array([np.nan]*len(psi_t)),
        np.array([np.nan]*len(psi_t)),
        w_t,
        dps
    ), tm
    

def nl_tm_vs_time():
    """
    Calculate tearing mode solution in strongly non-linear regime and plot
    the solution as a function of time.
    """
    m=2
    n=1
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e8, 10000)
    
    td_sol, tm0 = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )
    
    psi_t, w_t, delta_primes = td_sol.psi_t, td_sol.w_t, td_sol.delta_primes

    print(psi_t)

    fig, ax = plt.subplots(1, figsize=(4,3))
    ax2 = ax.twinx()
    
    ax.plot(times, psi_t, label='Normalised perturbed flux', color='black')

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")

    ax2.plot(times, w_t, label='Normalised island width', color='red')
    ax2.set_ylabel(r"Normalised island width ($w/a$)")
    ax2.yaxis.label.set_color('red')
    
    ax.set_yscale('log')
    ax2.set_yscale('log')
    ax.set_xscale('log')
    ax2.set_xscale('log')

    fig.tight_layout()
    #plt.show()
    fname = f"nl_tm_time_evo_(m,n,A)=({m},{n},{solution_scale_factor})"
    savefig(
        fname
    )
    dataclass_to_disk(fname, td_sol)
    plt.show()

def const_psi_approximation():
    """
    Calculate layer_width * Delta' and plot as a function of layer width
    to test the constant-psi approximation for a strongly non-linear solution.

    The constant-psi approximation breaks down if layer_width*Delta' ~ 1
    """
    m=2
    n=1
    lundquist_number = 1e8
    axis_q = 1.2
    solution_scale_factor = 1e-5

    times = np.linspace(0.0, 1e8, 10000)

    td_sol, tm0 = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    d_delta = td_sol.delta_primes * td_sol.w_t
    
    fig, ax = plt.subplots(1)
    
    ax.plot(td_sol.w_t, d_delta, color='black')
    
    ax.set_xlabel(r"Layer width")
    ax.set_ylabel(r"$\delta \Delta'$")
    
    fig.tight_layout()
    #plt.show()
    savefig(
        f"nl_const_psi_approx_(m,n,A)=({m},{n},{solution_scale_factor})"
    )
    plt.show()

def const_psi_q_sweep():
    """
    Find and plot the maximum layer_width*Delta' value for a tearing mode as a
    function of the on-axis safety factor.

    Used to determine whether the constant-psi approximation remains true
    over a range of on-axis safety factor values.
    """
    m=2
    n=1
    lundquist_number = 1e8
    q_rs = m/n
    axis_qs = np.linspace(q_rs/q(0.0)-1e-2, q_rs/q(1.0)+1e-2, 100)

    solution_scale_factor = 1e-5

    times = np.linspace(0.0, 1e8, 1000)

    d_delta_maxs = []

    for axis_q in axis_qs:
        td_sol, tm0 = solve_time_dependent_system(
            m, n, lundquist_number, axis_q, solution_scale_factor, times
        )

        d_delta = td_sol.delta_primes * td_sol.w_t

        d_delta_maxs.append(max(d_delta))

    fig, ax = plt.subplots(1)

    ax.plot(axis_qs, d_delta_maxs, color='black')

    ax.set_xlabel(r"On-axis safety factor")
    ax.set_ylabel(r"Maximum $\delta \Delta'$")

    fig.tight_layout()
    #plt.show()
    savefig(
        f"nl_const_psi_approx_q_sweep_(m,n,A)=({m},{n},{solution_scale_factor})"
    )
    plt.show()

def parabola(x, a, b, c):
    """
    Parabolic function.
    """
    return a*x**2 + b*x + c

def nl_parabola_coefficients(tm: TearingModeSolution,
                             mag_shear: float,
                             lundquist_number: float,
                             delta_prime_linear: float,
                             psi_0: float) -> Tuple[float, float, float]:
    """
    Coefficients of the algebraic solution to the strongly non-linear tearing
    mode in the small island limit.

    Parameters:
        tm: TearingModeSolution
            Solution to the perturbed flux in the outer region of the plasma.
        mag_shear: float
            Magnetic shear at the resonant surface.
        lundquist_number: float
            The Lundquist number.
        delta_prime_linear: float
            Discontinuity parameter of the outer perturbed flux solution in the
            linear regime.
        psi_0: float
            Perturbed flux at t=0 at the resonant surface.
    """
    c_0 = (tm.r_s**3 / (64*lundquist_number**2))\
        *mag_shear*delta_prime_linear**2
    c_1 = np.sqrt(psi_0) * (tm.r_s**3 * mag_shear)**0.5\
        * delta_prime_linear/(4*lundquist_number)
    c_2 = psi_0

    return c_0, c_1, c_2

def nl_parabola(tm: TearingModeSolution,
                mag_shear: float,
                lundquist_number: float,
                delta_prime_linear: float,
                psi_0: float,
                times: np.array):
    """
    Full algebraic solution to the strongly non-linear tearing
    mode in the small island limit.

    Parameters:
        tm: TearingModeSolution
            Solution to the perturbed flux in the outer region of the plasma.
        mag_shear: float
            Magnetic shear at the resonant surface.
        lundquist_number: float
            The Lundquist number.
        delta_prime_linear: float
            Discontinuity parameter of the outer perturbed flux solution in the
            linear regime.
        psi_0: float
            Perturbed flux at t=0 at the resonant surface.
        times: float
            Times over which to calculate the algebraic solution.
    """
    c_0, c_1, c_2 = nl_parabola_coefficients(
        tm,
        mag_shear,
        lundquist_number,
        delta_prime_linear,
        psi_0
    )

    new_times = times - times[0]

    return c_0*(new_times**2) + c_1*new_times + c_2

def nl_tm_small_w():
    """
    Numerically solve the tearing mode problem in the strongly non-linear regime
    starting with a small seeding island value.

    We then plot this solution as a function of time together with the algebraic
    solution found in the small island limit.
    """
    m=3
    n=2
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e7, 10000)

    psi_t, w_t, tm0, dps = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    psi0 = psi_t[0]
    dp = delta_prime(tm0)
    s = magnetic_shear(tm0.r_s, m, n)

    a_theory, b_theory, c_theory = nl_parabola_coefficients(
        tm0,
        s,
        lundquist_number,
        dp,
        psi0
    )

    print(
        f"""Theoretical fit at^2 + bt + c:
            a={a_theory}, b={b_theory}, c={c_theory}"""
    )

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(times, psi_t, label='Numerical solution', color='black')

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")

    # Returns highest power coefficients first
    fit, cov = curve_fit(
        parabola,
        times,
        psi_t,
        p0=(a_theory, b_theory, c_theory),
        bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        x_scale = (1.0/psi0, 1.0/psi0, 1.0/psi0),
        method='trf'
    )
    perr = np.sqrt(np.diag(cov))
    print("Coefs: ", fit)
    print("Errors: ", perr)
    poly = np.poly1d(fit)


    t_fit = (a_theory, b_theory, c_theory)
    ax.plot(
        times, parabola(times, *t_fit), label='Theoretical poly', linestyle='dotted',
        color='red'
    )
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend(prop={'size': 7})

    ls_data = np.sum((parabola(times, *fit) - psi_t)**2)
    ls_theory = np.sum((parabola(times, *t_fit) - psi_t)**2)

    print(ls_data, ls_theory)

    fig.tight_layout()
    #plt.show()
    savefig(f"nl_small_w_(m,n,A)=({m},{n},{solution_scale_factor})")
    plt.show()

def algebraic_departure():
    """
    Calculate the difference between the numerical solution to the non-linear
    equations and the algebraic solution found in the small island limit.
    """
    m=3
    n=2
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 5e7, 10000)

    psi_t, w_t, tm0, dps = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    psi0 = psi_t[0]
    dp = delta_prime(tm0)
    s = magnetic_shear(tm0.r_s, m, n)

    a_theory, b_theory, c_theory = nl_parabola_coefficients(
        tm0,
        s,
        lundquist_number,
        dp,
        psi0
    )

    print(
        f"""Theoretical fit at^2 + bt + c:
            a={a_theory}, b={b_theory}, c={c_theory}"""
    )

    fig, ax = plt.subplots(1, figsize=(4,3.5))
    #ax2 = ax.twinx()

    #ax.plot(times, psi_t, label='Numerical solution', color='black')

    #ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_xlabel(r"Numerical flux solution ($\delta \hat{\psi}^{(1)}$)")
    ax.set_ylabel(r"Fractional error in algebraic solution")

    #ax2.plot(times, w_t, label='Normalised island width', color='red')
    #ax2.set_ylabel(r"Normalised island width ($\hat{w}$)")
    #ax2.yaxis.label.set_color('red')

    # Returns highest power coefficients first
    fit, cov = curve_fit(
        parabola,
        times,
        psi_t,
        p0=(a_theory, b_theory, c_theory),
        bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        x_scale = (1.0/psi0, 1.0/psi0, 1.0/psi0),
        method='trf'
    )
    perr = np.sqrt(np.diag(cov))
    print("Coefs: ", fit)
    print("Errors: ", perr)
    poly = np.poly1d(fit)

    #ax.plot(
        #times, parabola(times, *fit), label='Order 2 poly fit', linestyle='dashed',
        #color='darkturquoise'
    #)

    t_fit = (a_theory, b_theory, c_theory)
    theoretical_psi = parabola(times, *t_fit)
    fractional_change = np.sqrt(((psi_t-theoretical_psi)/theoretical_psi)**2)

    print(fractional_change.shape)
    print(times.shape)
    print(max(times))
    #ax.plot(psi_t, fractional_change)
    ax.plot(psi_t, fractional_change)
    print(min(psi_t), max(psi_t))

    #ax2 = ax.twiny()
    #ax2.plot(psi_t, psi_t)
    #ax2.cla()

    #ax.plot(
        #times, parabola(times, *t_fit), label='Theoretical poly', linestyle='dotted',
        #color='red'
    #)
    ax.set_xscale('log')
    ax.set_yscale('log')

    #ax.legend(prop={'size': 7})

    ls_data = np.sum((parabola(times, *fit) - psi_t)**2)
    ls_theory = np.sum((parabola(times, *t_fit) - psi_t)**2)

    print(ls_data, ls_theory)

    fig.tight_layout()
    #plt.show()
    savefig(
        f"error_algebraic_sol_(m,n,A)=({m},{n},{solution_scale_factor})"
    )
    plt.show()

def marginal_stability(poloidal_mode: int = 2, toroidal_mode: int = 2):
    """
    Solve time dependent NL equation for multiple q-values. Plot
    final island width as a function of q(0)
    """
    m=poloidal_mode
    n=toroidal_mode
    lundquist_number = 1e8
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e8, 100)

    q_rs = m/n
    axis_qs = np.linspace(q_rs/q(0.0)-1e-2, q_rs/q(1.0)+1e-2, 100)

    final_widths = []

    for axis_q in axis_qs:
        psi_t, w_t, tm0, delta_primes = solve_time_dependent_system(
            m, n, lundquist_number, axis_q, solution_scale_factor, times
        )
        w_t = np.squeeze(w_t)

        saturation_width = np.mean(w_t[-20:])
        saturation_width_sem = sem(w_t[-20:])
        final_widths.append(
            (saturation_width, saturation_width_sem, delta_primes[0])
        )

    fig, ax = plt.subplots(1, figsize=(4,3))
    ax2 = ax.twinx()

    means, sems, delta_primes = zip(*final_widths)
    means = np.array(means)
    sems = np.squeeze(np.array(sems))

    ax.set_title(f"(m, n)=({m}, {n})")

    ax2.plot(axis_qs, delta_primes, color='red', alpha=0.5)
    ax.plot(axis_qs, means, color='black')
    ax2.set_ylabel(r"$a\Delta'$ at $t=0$", color='red')
    #ax2.hlines(
        #0.0, min(axis_qs), max(axis_qs), color='red', linestyle='--'
    #)
    #ax.fill_between(axis_qs, means-sems, means+sems, alpha=0.3)

    # Set ylim for delta' plot so that zeros align
    ax.set_ylim(bottom=-0.01)
    ax_bottom, ax_top = ax.get_ylim()
    ax2_bottom, ax2_top = ax2.get_ylim()
    ax2_bottom_new = ax2_top * (ax_bottom/ax_top)
    ax2.set_ylim(bottom=ax2_bottom_new)

    ax.set_xlabel("On-axis safety factor")
    ax.set_ylabel("Saturated island width $(w/a)$")

    ax.grid(which='both')
    fig.tight_layout()

    savefig(f"q_sweep_nl_(m,n,A)=({m},{n},{solution_scale_factor})")
    
    plt.show()

def marg_stability_multi_mode():
    """
    Create marginal stability plots for multiple modes.
    """
    modes = [
        (2,1),
        (2,2),
        (2,3),
        (3,1),
        (3,2),
        (3,3)
    ]
    for m,n in modes:
        marginal_stability(m, n)

if __name__=='__main__':
    nl_tm_vs_time()
    #nl_tm_small_w()
    #nl_tm_vs_time()
    #algebraic_departure()
    #const_psi_approximation()
    #const_psi_q_sweep()
