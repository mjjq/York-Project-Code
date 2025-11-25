"""
Implementation of equation 17 in Militello2004.
"""
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve

from tearing_mode_solver.outer_region_solver import (
    OuterRegionSolution, TearingModeParameters, delta_prime,
    solve_system, magnetic_shear, gamma_constant, growth_rate_full
)

def a_coefficient(j_prime_rs: float,
                  shear_rs: float,
                  poloidal_mode_number: int,
                  toroidal_mode_number: int,
                  inverse_aspect_ratio: float) -> float:
    """
    Calculate the a coefficient (below equation 5)

    Note: The paper automatically has all lengths normalised
    to minor radius. So R -> 1/epsilon

    Re-write this equation in terms of normalised parameters:

    a = -(m/n) * dJ_0/d(r/a) / (eps * s(r_s))

    :param j_prime_rs: Radial derivative in current density 
        profile at resonant surface
    :param shear_rs: Magnetic shear at the resonant surface
    :param poloidal_mode_number: Poloidal mode number
    :param toroidal_mode_number: Toroidal mode number
    :param inverse_aspect_ratio: Inverse aspect ratio of the tokamak
    """
    return -(
        poloidal_mode_number/toroidal_mode_number * j_prime_rs / 
        (inverse_aspect_ratio*shear_rs)
    )


def b_coefficient(r_s: float,
                  poloidal_mode_number: int,
                  toroidal_mode_number: int,
                  q_prime_rs: float,
                  q_double_prime_rs: float,
                  inverse_aspect_ratio: float,
                  shear_rs: float,
                  j_prime_rs: float,
                  j_double_prime_rs: float) -> float:
    """
    Calculate the b coefficient (below equation 5)

    :param r_s: Radius of rational surface
    :param poloidal_mode_number: Poloidal mode number
    :param toroidal_mode_number: Toroidal mode number
    :param q_prime_rs: First derivative in safety factor at r_s
    :param q_double_prime_rs: Second derivative in safety
        factor at r_s
    :param inverse_aspect_ratio: Inverse aspect ratio of the tokamak
    :param shear_rs: Magnetic shear at the resonant surface
    :param j_prime_rs: Radial derivative in current density 
        profile at resonant surface
    :param j_double_prime: Curvature of the current density
        profile at the resonant surface
    """
    a = a_coefficient(
        j_prime_rs, shear_rs, 
        poloidal_mode_number, toroidal_mode_number, 
        inverse_aspect_ratio
    )
    return 1.0/r_s * (
        a/(2.0*toroidal_mode_number) * r_s * q_double_prime_rs / q_prime_rs -
        poloidal_mode_number/(toroidal_mode_number*inverse_aspect_ratio*r_s*shear_rs) *
        (r_s**2.0 * j_double_prime_rs - r_s*j_prime_rs)
    )



def capital_a_coef(outer_sol: OuterRegionSolution) -> float:
    """
    Evaluate the A coefficient in equation 6.

    This can be derived by taking the derivative of equation
    6 for both the + and - solutions, evaluating these
    derivatives at x=0 (r=r_s), then summing the +/- solutions.

    This gives

    A = (psi_x^+ + psi_x^-)/psi(r_s)

    :param outer_sol: Outer region solution
    """
    first_term = (
        (outer_sol.dpsi_dr_backwards[-1] + outer_sol.dpsi_dr_forwards[-1]) /
        outer_sol.psi_forwards[-1]
    )

    return first_term

def f_coefficient(a_coef: float,
                  cap_a_coef: float,
                  delta: float,
                  b_coef: float) -> float:
    """
    Evaluate the F coefficient, equation 18.

    :param a_coef: a coefficient, see a_coefficient()
    :param cap_a_coef: A coefficient, see capital_a_coef()
    :param delta: Resistive layer width
    :param b_coef: b coefficient, see b_coefficient()
    """
    return (
        a_coef*cap_a_coef/2.0 +
        a_coef**2.0 * np.log(delta) -
        a_coef**2.0 * (alpha_2() + np.pi)/gamma_constant() + 
        b_coef
    )

@dataclass
class MilitelloCoefficients:
    a: float
    b: float
    capital_a: float
    delta_prime: float
    r_s: float
    k_c: float
    lundquist_number: float
    shear_rs: float
    poloidal_mode_number: int
    toroidal_mode_number: int

def calculate_coefficients(params: TearingModeParameters) -> MilitelloCoefficients:
    """
    Calculate all coefficients given in equations 17 and 18.

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

    rs, qs = zip(*params.q_profile)
    q_spline = UnivariateSpline(rs, qs, s=0)
    q_spline_prime = q_spline.derivative(1)
    q_spline_double_prime = q_spline.derivative(2)

    q_prime_rs = q_spline_prime(outer_sol.r_s)
    q_double_prime_rs = q_spline_double_prime(outer_sol.r_s)

    # Due to slightly different normalisations between our code
    # and militello's paper, we set the inverse aspect ratio
    # parameter to 1. This factor is probably already captured
    # in one of the other variables, and so including it here
    # leads to erroneous values for a and b.
    inv_aspect = 1.0 #params.r_minor/params.R0

    a_coef = a_coefficient(
        j_prime_rs, shear_rs, 
        params.poloidal_mode_number, params.toroidal_mode_number, 
        inv_aspect
    )

    b_coef = b_coefficient(
        outer_sol.r_s, 
        params.poloidal_mode_number, params.toroidal_mode_number, 
        q_prime_rs, q_double_prime_rs, inv_aspect, 
        shear_rs, j_prime_rs, j_double_prime_rs
    )

    cap_a = capital_a_coef(outer_sol)

    delta_prime_val = delta_prime(outer_sol)

    k_c = shear_rs * params.toroidal_mode_number

    return MilitelloCoefficients(
        a=a_coef,
        b=b_coef,
        capital_a=cap_a,
        delta_prime=delta_prime_val,
        r_s=outer_sol.r_s,
        k_c=k_c,
        lundquist_number=params.lundquist_number,
        shear_rs=shear_rs,
        poloidal_mode_number=params.poloidal_mode_number,
        toroidal_mode_number=params.toroidal_mode_number
    )

def alpha_2():
    """
    Solution to int_0^infty [z*xi - log(z^2 + 1)] dz ~ -0.378
    """
    return -0.378

def gamma_solvable(x: float,
                   coefs: MilitelloCoefficients) -> float:
    """
    Helper function for solving Millitello dispersion relation
    iteratively

    x: "Guess" value for the growth rate
    coefs: All coefficients needed to solve the dispersion relation
    """
    delta = (x/(coefs.lundquist_number * coefs.k_c**2))**(1/4) / coefs.r_s

    a1 = gamma_constant()

    f_coef = f_coefficient(
        coefs.a,
        coefs.capital_a,
        delta,
        coefs.b
    )

    return a1*x*delta*coefs.lundquist_number - coefs.delta_prime - a1*f_coef*delta

def growth_rate_militello(coefs: MilitelloCoefficients) -> float:
    """
    Evaluate Delta' with current density gradient and curvature 
    effects (Millitello 2004, eq 17)
    """
    initial_guess = 1e-5*growth_rate_full(
        coefs.poloidal_mode_number,
        coefs.toroidal_mode_number,
        coefs.lundquist_number,
        coefs.r_s,
        coefs.shear_rs,
        coefs.delta_prime
    )

    return fsolve(gamma_solvable, x0=initial_guess, args=coefs)[0]
