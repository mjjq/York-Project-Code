# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from linear_solver import TearingModeSolution, solve_system

@np.vectorize
def island_width(psi_rs: float,
                 r_s: float,
                 poloidal_mode: int,
                 toroidal_mode: int,
                 magnetic_shear: float) -> float:
    """
    Helical width of a magnetic island (maximum distance between separatrices
    of the magnetic island).
    """
    pre_factor = poloidal_mode*psi_rs/(toroidal_mode*magnetic_shear)
    if pre_factor >= 0.0:
        return 4.0*np.sqrt(pre_factor)
    
    return 0.0

def delta_prime_non_linear(tm: TearingModeSolution,
                           island_width: float,
                           epsilon: float = 1e-10) -> float:
    """
    Non-linear discontinuity parameter calculation using the solution of the
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
    dpsi_dr_min = tm.dpsi_dr_f_func(r_min)

    r_max = tm.r_s + island_width/2.0
    dpsi_dr_max = tm.dpsi_dr_b_func(r_max)

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
    m = poloidal_mode
    n = toroidal_mode
    axis_q = 1.0
    
    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)
    
    island_widths = np.linspace(0.0, 1.0, 100)
    
    delta_ps = [delta_prime_non_linear(tm, w) for w in island_widths]
    
    fig, ax = plt.subplots(1, figsize=(4,3))
    
    ax.plot(island_widths, delta_ps, label=f"(m,n)=({m},{n})")
    
    ax.set_xlabel(r"Normalised island width ($\hat{w}$)")
    ax.set_ylabel(r"$a\Delta ' (\hat{w})$")
    
    ax.hlines(
        0.0, xmin=0.0, xmax=1.0, color='red',
        linestyle='--', label=r"$a\Delta' = 0 $"
    )
    fig.tight_layout()

    ax.legend()

    #plt.show()
    plt.savefig(f"./output/island_saturation_(m,n)=({m},{n}).png", dpi=300)

    
if __name__=='__main__':
    island_saturation()
