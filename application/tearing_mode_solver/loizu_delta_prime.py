"""
Implementation of equation 3.1 in Loizu2023.

Assume sigma=0, i.e. flat resistivity profile.
"""
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import UnivariateSpline

from tearing_mode_solver.outer_region_solver import (
    OuterRegionSolution, TearingModeParameters, delta_prime,
    solve_system, magnetic_shear
)

def a_coefficient(j_prime_rs: float,
                  j_rs: float,
                  shear_rs: float) -> float:
    """
    Calculate the A coefficient (equation 3.2a)

    :param j_prime_rs: Radial derivative in current density 
        profile at resonant surface
    :param j_rs: Current density at the resonant surface
    :param shear_rs: Magnetic shear at the resonant surface
    """
    return j_prime_rs/j_rs * (1-2.0/shear_rs)


def b_coefficient(j_double_prime_rs: float,
                  j_rs: float,
                  shear_rs: float) -> float:
    """
    Calculate the B coefficient (equation 3.2b)

    :param j_double_prime: Curvature of the current density
        profile at the resonant surface
    :param j_rs: Current density at the resonant surface
    :param shear_rs: Magnetic shear at the resonant surface
    """
    return j_double_prime_rs/j_rs * (1-2.0/shear_rs)


def w0_coefficient(sigma_prime_coef: float,
                   a_coef: float) -> float:
    """
    Calculate the w0 coefficient (equation 3.2c), normalised
    to plasma minor radius a.

    :param sigma_prime: Value of equation 3.3 (see :func sigma_prime:)
    """
    return np.exp(-sigma_prime_coef/(2.0*a_coef))


def sigma_prime(outer_sol: OuterRegionSolution,
                a_coef: float) -> float:
    """
    Evaluate the Sigma' coefficient, eq 3.3.

    :param outer_sol: Outer region solution,
    :param a_coef: A coefficient, see :func a_coefficient:
    """
    epsilon = np.abs(outer_sol.r_range_bkwd[-1] - outer_sol.r_range_fwd[-1])

    first_term = (
        (outer_sol.dpsi_dr_backwards[-1] + outer_sol.dpsi_dr_forwards[-1]) /
        outer_sol.psi_forwards[-1]
    )

    second_term = -2.0*a_coef * (1+np.log(epsilon))

    return first_term+second_term

@dataclass
class LoizuCoefficients:
    a: float
    b: float
    sigma_prime: float
    w0: float
    delta_prime: float
    r_s: float

def calculate_coefficients(params: TearingModeParameters) -> LoizuCoefficients:
    """
    Calculate all coefficients given in equation 3.1.

    :param params: Tearing mode parameters
    """
    outer_sol = solve_system(params)

    rs, js = zip(*params.j_profile)
    j_spline = UnivariateSpline(rs, js, s=0)
    j_spline_prime = j_spline.derivative(1)
    j_spline_double_prime = j_spline.derivative(2)

    j_rs = j_spline(outer_sol.r_s)
    j_prime_rs = j_spline_prime(outer_sol.r_s)
    j_double_prime_rs = j_spline_double_prime(outer_sol.r_s)
    shear_rs = magnetic_shear(params.q_profile, outer_sol.r_s)

    a_coef = a_coefficient(j_prime_rs, j_rs, shear_rs)
    b_coef = b_coefficient(j_double_prime_rs, j_rs, shear_rs)
    sigma_p_coef = sigma_prime(outer_sol, a_coef)
    w0 = w0_coefficient(sigma_p_coef, a_coef)

    delta_prime_val = delta_prime(outer_sol)

    return LoizuCoefficients(
        a=a_coef,
        b=b_coef,
        sigma_prime=sigma_p_coef,
        w0=w0,
        delta_prime=delta_prime_val,
        r_s=outer_sol.r_s
    )


def delta_prime_loizu(w: float,
                      coefs: LoizuCoefficients) -> float:
    """
    Evaluate finite-island Delta' (Loizu equation 3.1).

    Note: Equation 3.1 includes factor of 1.22 as this is
    technically the rutherford equation. Hence, divide everything
    by 1.22 to get effective Delta'.

    Assume constant resistivity profile (lowercase sigma=0)
    """
    return coefs.delta_prime + (1.0/1.22)*w*(
        0.5*coefs.a**2 * np.log(w/coefs.w0) -
        2.21*coefs.a**2 + 0.40 * coefs.a/coefs.r_s + 
        0.5*coefs.b
    )