# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from linear_solver import TearingModeSolution, solve_system


def island_width(psi_rs: float,
                 r_s: float,
                 magnetic_shear: float) -> float:
    return 4.0*np.sqrt(psi_rs*r_s/magnetic_shear)

def delta_prime_non_linear(tm: TearingModeSolution,
                           island_width: float,
                           epsilon: float = 1e-10) -> float:
    """
    Non-linear rutherford equation calculation using the solution of the
    perturbed flux.

    Parameters
    ----------
    tm : TearingModeSolution
        Solution to the reduced MHD equation obtained from solve_system.
    island_width : float
        Width of the magnetic island.
    epsilon : float, optional
        The tolerance needed for the lower and upper tearing mode solutions to
        match. The default is 1e-10.

    Raises
    ------
    ValueError
        Raised if upper and lower TM solutions don't match.

    Returns
    -------
    float
        Linear delta' value.

    """
    psi_plus = tm.psi_backwards[-1]
    psi_minus = tm.psi_forwards[-1]
    
    if abs(psi_plus - psi_minus) > epsilon:
        raise ValueError(
            f"""Forwards and backward solutions 
            should be equal at resonant surface.
            (psi_plus={psi_plus}, psi_minus={psi_minus})."""
        )
    
    
    r_min = tm.r_s - island_width/2.0
    id_min = np.abs(tm.r_range_fwd - r_min).argmin()
    dpsi_dr_min = tm.dpsi_dr_forwards[id_min]
    
    r_max = tm.r_s + island_width/2.0
    id_max = np.abs(tm.r_range_bkwd - r_max).argmin()
    dpsi_dr_max = tm.dpsi_dr_backwards[id_max]

    delta_p = (dpsi_dr_max - dpsi_dr_min)/psi_plus

    return delta_p


def island_saturation():
    """
    Plot delta' as a function of the magnetic island width using the non-linear
    rutherford equation.

    Returns
    -------
    None.

    """
    poloidal_mode = 2
    toroidal_mode = 1
    axis_q = 1.0
    
    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)
    
    island_widths = np.linspace(0.0, 1.0, 100)
    
    delta_ps = [delta_prime_non_linear(tm, w) for w in island_widths]
    
    fig, ax = plt.subplots(1)
    
    ax.plot(island_widths, delta_ps)
    
    ax.set_xlabel("Normalised island width")
    ax.set_ylabel("$\hat{\Delta} ' (\hat{w})$")
    
    ax.hlines(0.0, xmin=0.0, xmax=1.0, color='red', linestyle='--')