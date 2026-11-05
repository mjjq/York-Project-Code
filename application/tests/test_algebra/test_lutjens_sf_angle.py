from dataclasses import dataclass

from algebraic.mercier_criterion.expand_mercier import (
    RZ_lutjens, calc_jacobian,
    calc_metric_tensor, calc_derivatives,
    grad_shafranov
)
from algebraic.mercier_criterion.rendering import(
    function_to_latex
)
from algebraic.mercier_criterion.mercier_integrals import (
    drhat_squared_dr, expected_rhat_squared, safety_factor
)
from algebraic.mercier_criterion.datatypes import (
    GradShafVariables, ToroidalCoordinates, LutjensShaping
)
from algebraic.mercier_criterion.straight_field_expansion import (
    expand_chi, expand_omega_e_eps, expand_omega_eps_sq
)
from algebraic.mercier_criterion.identities import (
    fast_trig_expand, fourier_coefficients, fourier_expression
)
import sympy as sp
from sympy import Rational
from sympy import Derivative

@dataclass
class LutjensChiTerms:
    chi_e: sp.Expr
    chi_eps: sp.Expr
    chi_eps_T: sp.Expr
    chi_eps_e: sp.Expr
    chi_eps_e_T: sp.Expr
    chi_eps_squared: sp.Expr
    chi_eps_squared_T: sp.Expr
    chi_total: sp.Expr

def lutjens_chi_terms(coords: ToroidalCoordinates,
                      shaping: LutjensShaping) -> LutjensChiTerms:
    D = shaping.D
    E = shaping.E
    T = shaping.T
    P = shaping.P
    Q = shaping.Q

    r = coords.r
    R0 = coords.R0
    omega = coords.omega
    e = coords.ellip
    eps = coords.epsilon

    chi_e = (Rational(1,2))*(E(r)/r - sp.diff(E(r),r)) * sp.sin(2*omega)
    chi_eps = (r/R0 + sp.diff(D(r), r))*sp.sin(omega)
    chi_eps_e = (
        (E(r)*sp.diff(D(r), r)/r - sp.diff(r*E(r), r)/(2*R0))*sp.sin(omega)+
        (E(r)-r*sp.diff(E(r), r))/(6*R0) * sp.sin(3*omega)
    ) 
    chi_eps_T = Rational(1,3) * (2*T(r)/r - sp.diff(T(r), r))*sp.sin(3*omega)
    chi_eps_e_T = -(
        E(r)*sp.diff(T(r), r) + 2*sp.diff(E(r), r)*T(r)
    )/r * sp.sin(omega)
    # Note we only have P and Q, no sum over Pn or Qn. In Lutjens,
    # the n=1 term in the sum exactly cancels the term proportional
    # to omega, so omit this from the check. 
    # (Note: sin[(n-1)w]/2(n-1)->w/2 for n->1)
    chi_eps_squared = (
        #omega * (r**2/(2*R0**2) + D(r)/R0 + r*sp.diff(D(r), r)/(2*R0)) +
        (r**2/(4*R0**2) + r*sp.diff(D(r), r)/(4*R0))*sp.sin(2*omega) +
        (
            ((P(r)-Q(r))/r - sp.diff(P(r)-Q(r),r)) * sp.sin(2*omega)/4
        )
    )
    chi_eps_squared_T = (
        (T(r)*sp.diff(D(r),r)/r - r*sp.diff(T(r),r)/(4*R0)) * sp.sin(2*omega) +
        (T(r)-r*sp.diff(T(r), r)/2) * sp.sin(4*omega)/(4*R0)
    )

    chi_total = (
        omega + e*chi_e + eps*chi_eps + e*eps*chi_eps_e +
        eps*chi_eps_T + eps*e*chi_eps_e_T + eps**2 * chi_eps_squared +
        eps**2 * chi_eps_squared_T
    )

    return LutjensChiTerms(
        chi_e=chi_e,
        chi_eps=chi_eps,
        chi_eps_T=chi_eps_T,
        chi_eps_e=chi_eps_e,
        chi_eps_e_T=chi_eps_e_T,
        chi_eps_squared=chi_eps_squared,
        chi_eps_squared_T=chi_eps_squared_T,
        chi_total=chi_total
    )

def test_lutjens_chi():
    """
    Test lutjens thesis eq 5.9 derivation.
    """
    coords, shaping = RZ_lutjens()
    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)


    lutjens_chi = lutjens_chi_terms(coords, shaping)
    chi_true = lutjens_chi.chi_total

    chi = expand_chi(coords, j)

    diff = sp.simplify(chi_true - chi)

    function_to_latex([chi, chi_true, diff])

    assert diff==0


def substitute_chi_terms(delta_chi_arguments: list[sp.Expr],
                         chi_terms_to_sub: list[sp.Expr],
                         coords: ToroidalCoordinates) -> sp.Expr:
    """
    Helper for substituting various chi terms in lutjens expansion.
    """
    chi_subbed = []

    dchi = sp.Symbol('delta_chi')
    for delta_chi_arg, chi_term in zip(delta_chi_arguments, chi_terms_to_sub):
        chi_term_expanded = chi_term.subs(coords.omega, coords.chi+dchi)
        chi_term_expanded = sp.expand_trig(chi_term_expanded)
        chi_term_expanded = fast_trig_expand(
            sp.expand(chi_term_expanded), dchi
        )
        chi_term_expanded = chi_term_expanded.subs(dchi, delta_chi_arg)
        chi_term_expanded = sp.expand(chi_term_expanded)
        chi_subbed.append(chi_term_expanded)

    return sum(chi_subbed) + coords.chi

def test_lutjens_omega():
    """
    Test lutjens thesis eq 5.17 and 5.18 derivations.
    """
    coords, shaping = RZ_lutjens()
    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)
    chi_expr = expand_chi(coords, j)

    omega_e_eps = expand_omega_e_eps(
        chi_expr, coords
    )
    omega_e_eps = fourier_expression(
        *fourier_coefficients(sp.expand(omega_e_eps), coords.chi),
        coords.chi
    )

    chi_t = lutjens_chi_terms(coords, shaping)
    
    chi = coords.chi
    e = coords.ellip
    eps = coords.epsilon
    omega = coords.omega

    # test e*epsilon form first, eq 5.17
    # Subtract chi from each argument in 5.17 as we expand in 
    # chi+delta_chi, where delta_chi~O(eps)+O(e)
    delta_chi_arguments = [
        - eps*(chi_t.chi_eps.subs(omega, chi) + chi_t.chi_eps_T.subs(omega, chi)),
        - e*chi_t.chi_e.subs(omega, chi),
        - e*chi_t.chi_e.subs(omega, chi),
        0,
        0
    ]
    chi_terms_to_sub = [
        -e*chi_t.chi_e,
        -eps*chi_t.chi_eps,
        -eps*chi_t.chi_eps_T,
        -eps*e*chi_t.chi_eps_e,
        -eps*e*chi_t.chi_eps_e_T
    ]
    chi_subbed = []

    omega_e_eps_true = substitute_chi_terms(
        delta_chi_arguments,
        chi_terms_to_sub,
        coords
    )

    omega_e_eps_true = fourier_expression(
        *fourier_coefficients(sp.expand(omega_e_eps_true), chi),
        chi
    )

    diff = sp.simplify(omega_e_eps - omega_e_eps_true)

    function_to_latex([
        sp.collect(omega_e_eps, eps), 
        sp.collect(omega_e_eps_true, eps), 
        sp.collect(diff, eps)
    ])

    assert diff==0