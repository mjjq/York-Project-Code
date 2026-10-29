import sympy as sp
from argparse import ArgumentParser
from copy import deepcopy

from algebraic.mercier_criterion.datatypes import (
    ToroidalCoordinates, 
    EPFLShaping, 
    LutjensShaping,
    Derivatives,
    MetricTensor,
    GradShafVariables,
    State
)
from algebraic.mercier_criterion.rendering import function_to_latex, summarise_to_latex
from algebraic.mercier_criterion.identities import (
    fourier_coefficients,
    fourier_expression
)

def expected_rhat_squared(coords: ToroidalCoordinates, 
                          shaping: LutjensShaping) -> sp.Expr:
    """
    Return equation 5.9 from Lutjens' thesis.

    This is an expression, so subtract RHS terms from both sides.

    To be compared with the output of drhat_squared_dr
    """
    r = coords.r
    e = coords.ellip
    D = shaping.D
    E = shaping.E
    T = shaping.T
    P = shaping.P
    Q = shaping.Q
    R0 = coords.R0

    lhs = P(r)*(1+e*E(r)/r) + Q(r)*(1-e*E(r)/r)
    rhs = r**3/(4*R0**2) + r*D(r)/R0 - e*r**2*E(r)/(2*R0**2) - e*E(r)*T(r)/R0

    return lhs - rhs


def drhat_squared_dr(jac: sp.Expr, 
                     coords: ToroidalCoordinates) -> sp.Expr:
    """
    Evaluate and expand derivative of straight field line radius in terms of
    r, omega. Equation 5.7 in Lutjens' thesis.

    I.e. d(rhat^2)/dr = R0/pi * oint_0^2pi [J/R^2] domega

    Then, extract the O(eps^2) term which itself contains terms up to O(e)

    The O(eps^2) term must equal zero for r=rhat, which thus constrains
    shaping.P and shaping.Q
    """
    j_over_r2 = sp.series(jac/coords.R**2/coords.epsilon**2, coords.epsilon, 0, 4).removeO()
    j_over_r2 = sp.series(j_over_r2, coords.ellip, 0, 4).removeO()
    first_int = sp.integrate(
        j_over_r2, 
        (coords.omega, 0, 2*sp.pi)
    )
    first_int = sp.series(first_int, coords.epsilon, 0, 4).removeO()
    first_int = sp.series(first_int, coords.ellip, 0, 4).removeO()

    first_int = sp.expand(first_int*coords.R0/(2*sp.pi))
    first_int = sp.collect(first_int, coords.epsilon)

    eps_squared_term = first_int.coeff(coords.epsilon**2)
    eps_squared_term = sp.series(eps_squared_term, coords.ellip, 0, 2).removeO()

    return eps_squared_term

def integral_5(coords: ToroidalCoordinates, jac: sp.Expr) -> sp.Expr:
    """
    Evaluate I_5 in A.2 (Lutjens 1992).
    """
    integral = sp.integrate(jac, (coords.omega, 0, 2*sp.pi))/(2*sp.pi)
    integral = sp.simplify(integral)

    function_to_latex(integral);exit()