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

# import warnings
# warnings.filterwarnings("error")

from linear_solver import (
    TearingModeSolution, solve_system, magnetic_shear,
    scale_tm_solution, delta_prime
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
    
    if psi[0]<0.0:
        print("Warning, negative flux. Setting to zero.")
        # psi[0]=0.0
    
    try:  
        s = mag_shear
        w = island_width(psi, tm.r_s, s)
        delta_prime = delta_prime_non_linear(tm, w)
        dpsi_dt = (0.25/lundquist_number)*\
            (np.sqrt((tm.r_s**3)*s*psi) * delta_prime)
    except RuntimeWarning:
        print(f"Invalid sqrt value: s={s}, psi={psi}, r_s={tm.r_s}")
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

    w_t = island_width(psi_t, tm.r_s, s)
    
    return np.squeeze(psi_t), w_t, tm
    

def nl_tm_vs_time():
    m=2
    n=1
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 0.01

    times = np.linspace(0.0, 1e8, 1000)
    
    psi_t, w_t, tm0 = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

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
    plt.savefig(
        f"./output/nl_tm_time_evo_(m,n,A)=({m},{n},{solution_scale_factor}).png", 
        dpi=300
    )
    plt.show()

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def nl_tm_small_w():
    m=2
    n=1
    resistivity = 0.0001
    axis_q = 1.0
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



if __name__=='__main__':
    #nl_tm_vs_time()
    #nl_tm_small_w()
    nl_tm_vs_time()
