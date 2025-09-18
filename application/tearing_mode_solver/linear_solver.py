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
    delta_prime_non_linear, island_width, gamma_constant,
    growth_rate
)
from tearing_mode_solver.helpers import (
    TimeDependentSolution, TearingModeParameters
)


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

    tm = solve_system(params)

    psi_t0 = params.initial_flux

    s = magnetic_shear(
        params.q_profile,
        tm.r_s
    )

    delta_p, gr = growth_rate(
        params.poloidal_mode_number,
        params.toroidal_mode_number,
        params.lundquist_number,
        params.q_profile,
        tm
    )
    exp_factor = gr*t_range
    # Avoid getting infinities by capping exponential factor to 100
    exp_factor[exp_factor > 100.0] = 100.0

    psi_t = np.exp(exp_factor + np.log(psi_t0))

    # We get weird numerical bugs sometimes returning large or nan values.
    # Set these to zero.

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
