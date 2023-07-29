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

from linear_solver import (
    TearingModeSolution, solve_system, growth_rate_scale, magnetic_shear
)
from non_linear_solver import delta_prime_non_linear, island_width


def flux_time_derivative(psi: float,
                         time: float,
                         tm: TearingModeSolution,
                         poloidal_mode: int,
                         toroidal_mode: int,
                         resistivity: float,
                         epsilon: float = 1e-5):

    m = poloidal_mode
    n = toroidal_mode

    s = magnetic_shear(tm.r_s, m, n)
    w = island_width(psi, tm.r_s, s)
    delta_prime = delta_prime_non_linear(tm, w)
    
    dpsi_dt = 0.25*resistivity*np.sqrt(s*psi/tm.r_s) * delta_prime
    
    return dpsi_dt


def solve_time_dependent_system(poloidal_mode: int, 
                                toroidal_mode: int, 
                                resistivity: float,
                                axis_q: float,
                                t_range: np.array = np.linspace(0.0, 1e5, 10)):
    
    tm = solve_system(poloidal_mode, toroidal_mode, axis_q, n=10000)

    psi_t0 = tm.psi_forwards[-1]
    
    results = odeint(
        flux_time_derivative,
        psi_t0,
        t_range,
        args = (tm, poloidal_mode, toroidal_mode, resistivity)
    )
    
    return results
    

def nl_tm_vs_time():
    m=3
    n=2
    resistivity = 1.0
    axis_q = 1.0

    times = np.linspace(0.0, 1e3, 100)
    
    psi_t = solve_time_dependent_system(
        m, n, resistivity, axis_q, times
    )

    fig, ax = plt.subplots(1, figsize=(4,3))
    
    ax.plot(times, psi_t, label='Normalised perturbed flux')
    
    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")  
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")
    ax.legend(prop={'size':8})
    fig.tight_layout()
    
    #plt.savefig(f"linear_tm_time_evo_(m,n)={m},{n}.png", dpi=300)
    

    
if __name__=='__main__':
    nl_tm_vs_time()
        
        