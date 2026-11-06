import sympy as sp
from argparse import ArgumentParser
from copy import deepcopy

from debug.log import logger

from algebraic.mercier_criterion.datatypes import (
    ToroidalCoordinates, 
    EPFLShaping, 
    LutjensShaping,
    Derivatives,
    MetricTensor,
    GradShafVariables,
    State
)
from algebraic.mercier_criterion.rendering import (
    function_to_latex, summarise_to_latex
)
from algebraic.mercier_criterion.identities import (
    fourier_coefficients,
    fourier_expression,
    fourier_rewrite,
    truncate,
    fast_trig_expand,
    fast_trig_expand_2,
    fast_truncate,
    faster_truncate,
    truncated_product_2d
)
import algebraic.mercier_criterion.mercier_integrals as integrals

def expand_chi(coords: ToroidalCoordinates, jacobian: sp.Expr) -> sp.Expr:
    """
    Expand the straight field line angle chi as a function of coords.r
    and coords.omega (expand lutjens thesis 5.7 and recover eq 5.15)
    """
    integral_indef = integrals.drhat_squared_dr_indefinite(
        jacobian, coords
    )
    logger.debug("Calculated indef integral")
    inv_integral = 1/integrals.j_over_rsquared_integral(
        jacobian, coords
    )
    logger.debug("Calculated chi integral")

    chi_of_omega = 2*sp.pi*integral_indef*inv_integral
    chi_of_omega = sp.series(chi_of_omega, coords.ellip, 0, 2).removeO()
    chi_of_omega = sp.series(chi_of_omega, coords.epsilon, 0, 3).removeO()
    #chi_of_omega = sp.collect(chi_of_omega, coords.epsilon)

    logger.debug("Expanded chi")

    return truncate(chi_of_omega, coords.epsilon, coords.ellip)


def expand_omega(chi_of_omega: sp.Expr, 
                 coords: ToroidalCoordinates,
                 order_epsilon: int,
                 order_ellip: int,
                 n_iter: int = 2) -> sp.Expr:
    """
    Invert chi expansion to get omega in terms of its own harmonics,
    i.e. Lutjens thesis 5.17 and 5.18

    n_iter is the number of iterations of successive substitutions.
    This is explained just above eq 5.17 in lutjens' thesis.
    """
    perturbed_part_of_chi = chi_of_omega - coords.omega

    perturbed_part_of_chi = sp.series(
        perturbed_part_of_chi, coords.ellip, 0, order_ellip
    ).removeO()
    perturbed_part_of_chi = sp.series(
        perturbed_part_of_chi, coords.epsilon, 0, order_epsilon
    ).removeO()

    omega_expanded = coords.chi-perturbed_part_of_chi

    delta_chi = sp.Symbol('delta_chi')
    

    for i in range(n_iter):
        omega_expanded = omega_expanded.subs(
            coords.omega, coords.chi+delta_chi
        )
        logger.debug("Sub in delta chi")
        omega_expanded = sp.expand_trig(omega_expanded)
        logger.debug("Separate delta chi in trig terms")
        omega_expanded = fast_trig_expand(
            sp.expand(omega_expanded), delta_chi, 2
        )
        logger.debug("Taylor expanded delta chi trig terms")
        omega_expanded = omega_expanded.subs(delta_chi, -perturbed_part_of_chi)
        logger.debug("Sub delta chi")
        omega_expanded = sp.series(
            omega_expanded, coords.ellip, 0, order_ellip
        ).removeO()
        omega_expanded = sp.series(
            omega_expanded, coords.epsilon, 0, order_epsilon
        ).removeO()
        logger.debug(f"Expanded omega, iteration {i}")

    #function_to_latex(omega_expanded); exit()
    #omega_expanded = sp.simplify(omega_expanded)
    omega_expanded = sp.collect(
        sp.series(omega_expanded, coords.epsilon, 0, 3), 
        coords.epsilon
    )

    return omega_expanded

def sub_omega_expr(expr: sp.Expr,
                   omega_expanded: sp.Expr,
                   coords: ToroidalCoordinates,
                   order_epsilon: int = 3,
                   order_ellip: int = 2) -> sp.Expr:
    """
    Expand expr(r,omega) into expr(r,chi) using the expansion
    for omega(r, chi) 
    """

    delta_chi = sp.Symbol('delta_chi')

    expr_expanded = expr.subs(
            coords.omega, coords.chi+delta_chi
        )
    logger.debug("Sub in delta chi")

    logger.debug("Separate delta chi in trig terms")
    expr_expanded = fast_trig_expand_2(
        expr_expanded, delta_chi, 3
    )
    logger.debug("Taylor expanded delta chi trig terms")
    perturbed_part_of_omega = omega_expanded - coords.chi
    perturbed_part_of_omega = fourier_rewrite(
        perturbed_part_of_omega, coords.chi
    )
    poly = sp.Poly(expr_expanded, delta_chi)

    summed = 0
    for (power,), coeff in poly.terms():
        dchi_replacement = sp.expand(perturbed_part_of_omega**power)
        trunc_prod = truncated_product_2d(
            dchi_replacement, coeff,
            coords.epsilon, coords.ellip, 2, 1
        )
        summed += trunc_prod


    #expr_expanded = expr_expanded.subs(delta_chi, perturbed_part_of_omega)
    logger.debug("Sub delta chi")
    #function_to_latex(summed); exit()

    # expr_expanded = faster_truncate(expr_expanded, coords.ellip, 2)
    # expr_expanded = faster_truncate(expr_expanded, coords.epsilon, 3)
    # expr_expanded = sp.collect(expr_expanded, coords.epsilon)

    return summed



def expand_omega_e_eps(chi_of_omega: sp.Expr, 
                       coords: ToroidalCoordinates) -> sp.Expr:
    """
    Expand omega to order e*epsilon
    """
    return expand_omega(
        chi_of_omega, coords,
        2, 2
    )

def expand_omega_eps_sq(chi_of_omega: sp.Expr, 
                        coords: ToroidalCoordinates) -> sp.Expr:
    """
    Expand omega to order epsilon^2 + e^0
    """
    return expand_omega(
        chi_of_omega, coords,
        3,1
    )

def combine_omega_expressions(omega_e_eps: sp.Expr, 
                              omega_eps_squared: sp.Expr, 
                              coords: ToroidalCoordinates):
    eps = coords.epsilon

    a = sp.expand(omega_e_eps)        # valid through eps^1 (all e^0, e^1 terms)
    b = sp.expand(omega_eps_squared)  # valid through eps^2 (e^0 only)

    # b's eps^0 and eps^1 terms are already in a, so take only b's eps^2 part
    b_eps2 = eps**2 * b.coeff(eps, 2)

    omega_total = a + b_eps2

    # sanity check: the eps^1 part must come from a alone
    assert sp.simplify(
        omega_total.coeff(eps, 1) - a.coeff(eps, 1)
    ) == 0

    return omega_total