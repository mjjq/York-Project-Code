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
    truncate
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

    inv_integral = sp.series(inv_integral, coords.ellip, 0, 2).removeO()
    inv_integral = sp.series(inv_integral, coords.epsilon, 0, 3).removeO()

    chi_of_omega = 2*sp.pi*integral_indef*inv_integral
    chi_of_omega = sp.series(chi_of_omega, coords.ellip, 0, 2).removeO()
    chi_of_omega = sp.series(chi_of_omega, coords.epsilon, 0, 3).removeO()
    chi_of_omega = sp.collect(sp.expand(chi_of_omega), coords.epsilon)

    logger.debug("Expanded chi")

    return truncate(chi_of_omega, coords.epsilon, coords.ellip)


def expand_omega(chi_of_omega: sp.Expr, 
                 coords: ToroidalCoordinates,
                 n_iter: int = 2) -> sp.Expr:
    """
    Invert chi expansion to get omega in terms of its own harmonics,
    i.e. Lutjens thesis 5.17 and 5.18

    n_iter is the number of iterations of successive substitutions.
    This is explained just above eq 5.17 in lutjens' thesis.
    """
    perturbed_part_of_chi = chi_of_omega - coords.omega

    perturbed_part_of_chi = sp.series(
        perturbed_part_of_chi, coords.epsilon, 0, 2
    ).removeO()
    perturbed_part_of_chi = sp.series(
        perturbed_part_of_chi, coords.ellip, 0, 2
    ).removeO()

    omega_expanded = coords.chi-perturbed_part_of_chi

    delta_chi = sp.Symbol('delta_chi')
    


    for i in range(n_iter):
        omega_expanded = omega_expanded.subs(
            coords.omega, coords.chi+delta_chi
        )
        omega_expanded = sp.expand_trig(omega_expanded)
        omega_expanded = sp.series(omega_expanded, delta_chi, 0, 2).removeO()
        omega_expanded = omega_expanded.subs(delta_chi, perturbed_part_of_chi)
        omega_expanded = sp.series(
            omega_expanded, coords.epsilon, 0, 3
        ).removeO()
        logger.debug(f"Expanded omega, iteration {i}")

    omega_expanded = sp.series(
            omega_expanded, coords.ellip, 0, 2
    ).removeO()
    omega_expanded = omega_expanded = sp.series(
            omega_expanded, coords.epsilon, 0, 3
    ).removeO()
    #function_to_latex(omega_expanded); exit()
    # omega_expanded = fourier_expression(
    #     *fourier_coefficients(omega_expanded, coords.chi), 
    #     coords.chi
    # )

    return omega_expanded