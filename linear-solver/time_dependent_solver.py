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

from linear_solver import (
    TearingModeSolution, solve_system, growth_rate_scale
)

def flux_time_derivative(psi: np.array,
                         time: float,
                         r_range_fwd: np.array,
                         r_range_bkwd: np.array,
                         K: float,
                         epsilon: float = 1e-5):
    
    psi_f, psi_b = psi.reshape(2, len(psi)//2)
    
    dr_fwd = r_range_fwd[-1] - r_range_fwd[-2]
    dpsi_dr_forwards = (psi_f[-1] - psi_f[-2]) / dr_fwd
    
    dr_bkwd = r_range_bkwd[-1] - r_range_bkwd[-2]
    dpsi_dr_backwards = (psi_b[-1] - psi_b[-2]) / dr_bkwd

    psi_rs_f = psi_f[-1]
    psi_rs_b = psi_b[-1]
    
    if(abs(psi_rs_f-psi_rs_b) > epsilon):
        print("Warning, fluxes at resonant surface do not match!")
    
    delta_prime = complex((dpsi_dr_backwards-dpsi_dr_forwards)/psi_rs_f)
    
    dpsi_dt_forwards = (K*psi_f*(delta_prime)**(4/5)).real
    dpsi_dt_backwards = (K*psi_b*(delta_prime)**(4/5)).real
    
    return np.concatenate((dpsi_dt_forwards, dpsi_dt_backwards))


def solve_time_dependent_system(poloidal_mode: int, 
                                toroidal_mode: int, 
                                lundquist_number: float,
                                axis_q: float = 1.0):
    
    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)
    grs = growth_rate_scale(
        lundquist_number, tm.r_s, poloidal_mode, toroidal_mode
    )
    
    t_range = np.linspace(0.0, 0.1, 3)
    print(t_range)

    y0 = np.concatenate((tm.psi_forwards, tm.psi_backwards))
    
    flux_time_derivative(
        y0,
        t_range[0],
        tm.r_range_fwd,
        tm.r_range_bkwd,
        grs
    )
    
    results = odeint(
        flux_time_derivative,
        y0,
        t_range,
        args = (tm.r_range_fwd, tm.r_range_bkwd, grs)
    )
    
    print(results.shape)
    
    results_forward, results_backward = results.reshape(
        2, len(t_range), len(y0)//2
    )
    
    return results_forward, results_backward, tm
    
    
if __name__=='__main__':
    m=3
    n=2
    lundquist_number = 1e8
    
    res_f, res_b, tm = solve_time_dependent_system(m, n, lundquist_number)
    
    fig, ax = plt.subplots(1)
    
    for r_f in res_f:
        ax.plot(tm.r_range_fwd, r_f)
    for r_b in res_b:
        ax.plot(tm.r_range_bkwd, r_b)