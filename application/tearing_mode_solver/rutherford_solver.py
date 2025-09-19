#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:06:24 2023

@author: marcus
"""
import numpy as np
from typing import Tuple
from scipy.integrate import odeint


from tearing_mode_solver.outer_region_solver import (
    OuterRegionSolution, solve_system, magnetic_shear,
    delta_prime_non_linear, island_width, gamma_constant
)
from tearing_mode_solver.helpers import (
    TimeDependentSolution, TearingModeParameters
)


def flux_time_derivative(psi: float,
                         time: float,
                         tm: OuterRegionSolution,
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
        tm: OuterRegionSolution
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

    pre_factor = 2.0**(5.0/4.0)/gamma_constant()

    if sqrt_factor >= 0.0:
        dpsi_dt = (
            0.5*pre_factor*((n*s/m)**0.5)*(psi**0.5)*tm.r_s**2 *
            (delta_prime/lundquist_number)
        )

    else:
        dpsi_dt = 0.0

    return dpsi_dt


def solve_time_dependent_system(params: TearingModeParameters,
                                t_range: np.array = np.linspace(0.0, 1e5, 10))\
        -> Tuple[TimeDependentSolution, OuterRegionSolution]:
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
            Array of time values to record. Each element will have an
            associated perturbed flux, derivative etc calculated for that time.
    """

    tm = solve_system(params, resolution=1e-5, r_s_thickness=1e-7)

    psi_t0 = params.initial_flux

    s = magnetic_shear(
        params.q_profile,
        tm.r_s
    )

    psi_t = odeint(
        flux_time_derivative,
        psi_t0,
        t_range,
        args=(tm, params.poloidal_mode_number, params.toroidal_mode_number,
              params.lundquist_number, s)
    )

    # We get weird numerical bugs sometimes returning large or nan values.
    # Set these to zero.
    psi_t[np.abs(psi_t) > 1e10] = 0.0
    psi_t[np.argwhere(np.isnan(psi_t))] = 0.0

    w_t = np.squeeze(
        island_width(psi_t, tm.r_s, params.poloidal_mode_number,
                     params.toroidal_mode_number, s)
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
