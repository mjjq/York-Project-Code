from typing import Tuple
import numpy as np

from tearing_mode_solver.outer_region_solver import OuterRegionSolution, \
    solve_system, magnetic_shear, delta_prime
from tearing_mode_solver.helpers import TearingModeParameters


def parabola(x, a, b, c):
    """
    Parabolic function.
    """
    return a*x**2 + b*x + c



def nl_parabola_coefficients(tm: OuterRegionSolution,
                             mag_shear: float,
                             lundquist_number: float,
                             delta_prime_linear: float,
                             psi_0: float) -> Tuple[float, float, float]:
    """
    Coefficients of the algebraic solution to the strongly non-linear tearing
    mode in the small island limit.

    Parameters:
        tm: OuterRegionSolution
            Solution to the perturbed flux in the outer region of the plasma.
        mag_shear: float
            Magnetic shear at the resonant surface.
        lundquist_number: float
            The Lundquist number.
        delta_prime_linear: float
            Discontinuity parameter of the outer perturbed flux solution in the
            linear regime.
        psi_0: float
            Perturbed flux at t=0 at the resonant surface.
    """
    c_0 = (tm.r_s**3 / (64*lundquist_number**2))\
        *mag_shear*delta_prime_linear**2
    c_1 = np.sqrt(psi_0) * (tm.r_s**3 * mag_shear)**0.5\
        * delta_prime_linear/(4*lundquist_number)
    c_2 = psi_0

    return c_0, c_1, c_2

def nl_parabola(tm: OuterRegionSolution,
                mag_shear: float,
                lundquist_number: float,
                delta_prime_linear: float,
                psi_0: float,
                times: np.array):
    """
    Full algebraic solution to the strongly non-linear tearing
    mode in the small island limit.

    Parameters:
        tm: OuterRegionSolution
            Solution to the perturbed flux in the outer region of the plasma.
        mag_shear: float
            Magnetic shear at the resonant surface.
        lundquist_number: float
            The Lundquist number.
        delta_prime_linear: float
            Discontinuity parameter of the outer perturbed flux solution in the
            linear regime.
        psi_0: float
            Perturbed flux at t=0 at the resonant surface.
        times: float
            Times over which to calculate the algebraic solution.
    """
    c_0, c_1, c_2 = nl_parabola_coefficients(
        tm,
        mag_shear,
        lundquist_number,
        delta_prime_linear,
        psi_0
    )

    new_times = times - times[0]

    return c_0*(new_times**2) + c_1*new_times + c_2

def get_parab_coefs(params: TearingModeParameters,
                    initial_nl_flux: float) -> Tuple[float, float, float]:
    """
    Calculate the parabola coefficients using a TearingModeParameters 
    object and an initial (non-linear) flux. Note: This is different from the
    initial flux specified in params.

    Parameters
    ----------
    params : TearingModeParameters
        Parameters of the tearing mode in question.
    initial_nl_flux : float
        Initial flux in the non-linear regime (not the same as initial flux of
        the model simulation).

    Returns
    -------
    Tuple[float, float, float].
        c_0: t^2 coefficient of parabola
        c_1: t coefficient of parabola
        c_2: Constant coefficient of parabola

    """
    outer_sol = solve_system(params)

    s = magnetic_shear(params.q_profile, outer_sol.r_s)
    dp = delta_prime(outer_sol)

    return nl_parabola_coefficients(
        outer_sol,
        s,
        params.lundquist_number,
        dp,
        initial_nl_flux
    )
