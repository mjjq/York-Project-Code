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
    GradShafVariables
)
from algebraic.mercier_criterion.straight_field_expansion import (
    expand_chi
)
import sympy as sp
from sympy import Derivative

def test_rhat_squared_equation():
    """
    Test lutjens thesis eq 5.9 derivation.

    This test will fail as we appear to recover an extra term
    of -2TT' in our derivation of the r constraint.

    The presence of this factor in our derivation is consistent
    with Jon's derivation at zero ellipticity (EPFL slides, lecture 2, slide 37)

    I therefore believe this to be an error in Hinrich's equation.
    """
    from algebraic.mercier_criterion.expand_mercier import gs_simplify_lutjens
    from algebraic.mercier_criterion.datatypes import State
    state_filename = "expand_mercier_state.pkl"
    
    coords, shaping = RZ_lutjens()
    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)
    gs = GradShafVariables(
        None, None, None, None, None, None, None, None
    )

    state = State(
        coords, shaping, derivatives,
        metric, gs, j
    )


    D = state.shaping.D
    E = state.shaping.E
    T = state.shaping.T
    P = state.shaping.P
    Q = state.shaping.Q

    r = state.coords.r
    R0 = state.coords.R0
    omega = state.coords.omega
    e = state.coords.ellip
    eps = state.coords.epsilon

    chi_e = (1/2)*(E(r)/r - sp.diff(E(r),r)) * sp.sin(2*omega)
    chi_eps = (r/R0 + sp.diff(D(r), r))*sp.sin(omega)
    chi_eps_e = (
        (E(r)*sp.diff(D(r), r)/r - sp.diff(r*E(r), r)/(2*R0))*sp.sin(omega)+
        (E(r)-r*sp.diff(E(r), r))/(6*R0) * sp.sin(3*omega)
    ) 
    chi_eps_T = 1/3 * (2*T(r)/r - sp.diff(T(r), r))*sp.sin(3*omega)
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

    chi_true = (
        omega + e*chi_e + eps*chi_eps + e*eps*chi_eps_e +
        eps*chi_eps_T + eps*e*chi_eps_e_T + eps**2 * chi_eps_squared +
        eps**2 * chi_eps_squared_T
    )

    chi = expand_chi(state.coords, state.jacobian)

    diff = sp.simplify(chi_true - chi)

    function_to_latex([chi, chi_true, diff])

    assert diff==0