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
    OuterRegionSolution, solve_system, growth_rate_scale
)

def flux_time_derivative(psi: np.array,
                         time: float,
                         r_range_fwd: np.array,
                         r_range_bkwd: np.array,
                         K: float,
                         epsilon: float = 1e-5):
    """
    Calculate first order time derivative of the perturbed flux
    in the inner region of the tearing mode using the linear time-dependent
    differential equation.

    This is passed to scipy's ODE function to be integrated.

    This function manually calculates the discontinuity parameter Delta'
    instead of using recently created functions. TODO: Update
    implementation to use full OuterRegionSolution and faster Delta' functions.

    Parameters:
        psi: np.array
            Full outer solution to the perturbed flux concatenated into a single
            1D array.
        time: float
            The current time of the simulation.
        r_range_fwd: np.array
            Radial co-ordinate values associated with the forward outer solution
        r_range_bkwd: np.array
            Radial co-ordinate values associated with the backward outer
            solution
        K: float
            Growth rate multiplier for the current tearing mode. See
            growth_rate_scale() in linear_solver
    """

    psi_f, psi_b = psi.reshape(2, len(psi)//2)
    
    dr_fwd = r_range_fwd[-1] - r_range_fwd[-2]
    dpsi_dr_forwards = np.gradient(psi_f, dr_fwd, edge_order=2)[-1]
    
    dr_bkwd = r_range_bkwd[-1] - r_range_bkwd[-2]
    dpsi_dr_backwards = np.gradient(psi_b, dr_bkwd, edge_order=2)[-1]

    psi_rs_f = psi_f[-1]
    psi_rs_b = psi_b[-1]
    
    if(abs(psi_rs_f-psi_rs_b) > epsilon):
        print("Warning, fluxes at resonant surface do not match!")
    
    delta_prime = complex((dpsi_dr_backwards-dpsi_dr_forwards)/psi_rs_f)
    
    dpsi_dt_forwards = (K*psi_f*(delta_prime)**(4/5)).real
    dpsi_dt_backwards = (K*psi_b*(delta_prime)**(4/5)).real
    
    ret = np.concatenate((dpsi_dt_forwards, dpsi_dt_backwards))
    
    return ret


def solve_time_dependent_system(poloidal_mode: int, 
                                toroidal_mode: int, 
                                lundquist_number: float,
                                axis_q: float = 1.0,
                                t_range: np.array = np.linspace(0.0, 1e5, 10)):
    """
    Numerically integrate the linear flux time derivative of a tearing
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
        t_range: np.array
            Array of time values to record. Each element will have an associated
            perturbed flux, derivative etc calculated for that time.
    """

    # TODO: Implementation of solve_system has changed. Need to update
    # the implementation here so that it's compatible with this.
    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)
    grs = growth_rate_scale(
        lundquist_number, tm.r_s, poloidal_mode, toroidal_mode
    )


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
    
    
    results_forwards = []
    results_backwards = []
    
    # Due to a weird numpy reshape bug, the rows of results_forward contains
    # results_backwards data every odd row and vice versa.
    # We work around this by manually iterating through each timestamp and
    # appending forwards and backwards solutions to the relevant variables.
    for psi_t in results:
        psi_t_fwd, psi_t_bkwd = psi_t.reshape(2, len(y0)//2)
        results_forwards.append(psi_t_fwd)
        results_backwards.append(psi_t_bkwd)
        
    return results_forwards, results_backwards, tm, t_range
    

def linear_tm_growth_plots():
    """
    Plot the full outer solution as a function of minor radius at different
    times using the linear time-dependent solver.
    """
    m=3
    n=2
    lundquist_number = 1e8
    
    times = np.linspace(0.0, 1e4, 3)
    
    res_f, res_b, tm, t_range = solve_time_dependent_system(
        m, n, lundquist_number,1.0, times
    )

    fig, ax = plt.subplots(1, figsize=(4,3))
    
    for i, psi_f in enumerate(res_f):
        psi_b = res_b[i]
        
        psi = np.concatenate((psi_f, psi_b[::-1]))
        r = np.concatenate((tm.r_range_fwd, tm.r_range_bkwd[::-1]))
        
        max_num = np.max((psi_b, psi_f))
        #print(max_num)
        
        ax.plot(r, psi, label=r'$\bar{\omega}_A t$='+f'{times[i]:.1e}')
        
    ax.vlines(
        tm.r_s, 0.0, np.max((res_f, res_b)), color='red', linestyle='--',
        label='$\hat{r}_s$='+f'{tm.r_s:.2f}'
    )
    
    ax.set_xlabel("Normalised minor radial co-ordinate $\hat{r}$")  
    ax.set_ylabel("Normalised perturbed flux $\delta \hat{\psi}^{(1)}$")
    ax.legend(prop={'size':8})
    fig.tight_layout()
    
    plt.savefig(f"linear_tm_time_evo_(m,n)={m},{n}.png", dpi=300)
    
    
def linear_tm_amplitude_vs_time():
    """
    Plot amplitude of the flux at the resonant surface as a function of time
    for a linear tearing mode.
    """
    m=4
    n=2
    lundquist_number = 1e8
    
    res_f, res_b, tm, t_range = solve_time_dependent_system(
        m, n, lundquist_number, 1.0, np.linspace(0.0, 1e5, 100)
    )

    fig, ax = plt.subplots(1, figsize=(4,3))
    
    res_amplitudes = [psi_f[-1] for psi_f in res_f]
    
    ax.set_yscale('log')
    ax.scatter(t_range, res_amplitudes, s=1)
    
    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(
        """Normalised perturbed flux at 
        resonant surface [$\delta \hat{\psi}^{(1)}(r_s)$]"""
    )
    
    fig.tight_layout()
        
    plt.savefig(f"res_amplitude_vs_time_(m,n)={m},{n}.png", dpi=300)
        
    dt = t_range[-1] - t_range[-2]
    dpsi_dt = np.gradient(res_amplitudes, dt)
    growth_rate = dpsi_dt/res_amplitudes

    growth_rate_clipped = growth_rate[1:-2]
    print(f"""Average growth_rate = {np.mean(growth_rate_clipped)} 
          +/- {sem(growth_rate_clipped)}""")
    
    fig2, ax2 = plt.subplots(1, figsize=(4,3))
    ax2.plot(t_range, growth_rate)
    
    ax2.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax2.set_ylabel(r"Normalised growth rate ($\gamma/\bar{\omega}_A$)")
    
    fig2.tight_layout()
    
if __name__=='__main__':
    linear_tm_growth_plots()
    #linear_tm_amplitude_vs_time()
        
        
