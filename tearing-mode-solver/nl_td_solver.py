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



from linear_solver import (
    TearingModeSolution, solve_system, magnetic_shear,
    scale_tm_solution, delta_prime, q
)
from non_linear_solver import delta_prime_non_linear, island_width
from pyplot_helper import savefig


def flux_time_derivative(psi: float,
                         time: float,
                         tm: TearingModeSolution,
                         poloidal_mode: int,
                         toroidal_mode: int,
                         lundquist_number: float,
                         mag_shear: float,
                         epsilon: float = 1e-5):

    
    m = poloidal_mode
    n = toroidal_mode
    
    #if psi[0]<0.0:
    #    print(f"Warning, negative flux at {time}. Setting to zero.")
    #    psi[0]=0.0
    # if psi[0]>1e-5:
    #     print("Warning, weirdly high psi value")
    # if np.isnan(psi[0]):
    #     print("Warning, psi is nan")
      
    s = mag_shear
    w = island_width(psi, tm.r_s, s)
    delta_prime = delta_prime_non_linear(tm, w)
    sqrt_factor = (tm.r_s**3)*s*psi
    
    
    if sqrt_factor >= 0.0:
        dpsi_dt = (0.25/lundquist_number)*\
            (np.sqrt(sqrt_factor) * delta_prime)

    else:
        dpsi_dt = 0.0
        
    
    return dpsi_dt


def solve_time_dependent_system(poloidal_mode: int, 
                                toroidal_mode: int, 
                                lundquist_number: float,
                                axis_q: float,
                                initial_scale_factor: float = 1.0,
                                t_range: np.array = np.linspace(0.0, 1e5, 10)):
    
    tm = solve_system(poloidal_mode, toroidal_mode, axis_q, n=10000)
    tm = scale_tm_solution(tm, initial_scale_factor)

    psi_t0 = tm.psi_forwards[-1]
    
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

    w_t = np.squeeze(island_width(psi_t, tm.r_s, s))

    dps = [delta_prime_non_linear(tm, w) for w in w_t]
    
    return np.squeeze(psi_t), w_t, tm, dps
    

def nl_tm_vs_time():
    m=3
    n=2
    lundquist_number = 1e8
    axis_q = 0.6188889
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e8, 200)
    
    psi_t, w_t, tm0 = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )
    
    print(psi_t)

    fig, ax = plt.subplots(1, figsize=(4,3))
    ax2 = ax.twinx()
    
    ax.plot(times, psi_t, label='Normalised perturbed flux', color='black')

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")

    ax2.plot(times, w_t, label='Normalised island width', color='red')
    ax2.set_ylabel(r"Normalised island width ($\hat{w}$)")
    ax2.yaxis.label.set_color('red')

    fig.tight_layout()
    #plt.show()
    savefig(
        f"nl_tm_time_evo_(m,n,A)=({m},{n},{solution_scale_factor})"
    )
    plt.show()

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def nl_tm_small_w():
    m=2
    n=1
    resistivity = 0.0001
    axis_q = 0.5
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e1, 100000)

    psi_t, w_t, tm0 = solve_time_dependent_system(
        m, n, resistivity, axis_q, solution_scale_factor, times
    )

    psi0 = psi_t[0]
    dp = delta_prime(tm0)
    s = magnetic_shear(tm0.r_s, m, n)

    a_theory = (resistivity/8)**2 * (s/tm0.r_s) * dp**2
    b_theory = np.sqrt(psi0) * (resistivity/4) * np.sqrt(s/tm0.r_s) * dp
    c_theory = psi0

    print(
        f"""Theoretical fit at^2 + bt + c:
            a={a_theory}, b={b_theory}, c={c_theory}"""
    )

    fig, ax = plt.subplots(1, figsize=(4,3))
    #ax2 = ax.twinx()

    ax.plot(times, psi_t, label='Numerical solution', color='black')

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")

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

    ax.plot(
        times, parabola(times, *fit), label='Order 2 poly fit', linestyle='dashed',
        color='darkturquoise'
    )

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

def marginal_stability():
    """
    Solve time dependent NL equation for multiple q-values. Plot
    final island width as a function of q(0)
    """
    m=3
    n=3
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



if __name__=='__main__':
    #nl_tm_vs_time()
    #nl_tm_small_w()
    marginal_stability()
