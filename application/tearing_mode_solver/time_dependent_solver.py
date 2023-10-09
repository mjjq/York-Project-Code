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

from tearing_mode_solver.outer_region_solver import (
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

    #psi_f, psi_b = psi.reshape(2, len(psi)//2)
    
    psi_f = psi[:len(r_range_fwd)]
    psi_b = psi[len(r_range_fwd):]


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
        #psi_t_fwd, psi_t_bkwd = psi_t.reshape(2, len(y0)//2)
        psi_t_fwd = psi_t[0:len(tm.r_range_fwd)]
        psi_t_bkwd = psi_t[len(tm.r_range_fwd):]
        results_forwards.append(psi_t_fwd)
        results_backwards.append(psi_t_bkwd)
        
    return results_forwards, results_backwards, tm, t_range
    