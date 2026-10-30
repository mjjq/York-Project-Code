from algebraic.mercier_criterion.expand_mercier import (
    RZ_lutjens, calc_jacobian,
    calc_metric_tensor, calc_derivatives,
    grad_shafranov
)
from algebraic.mercier_criterion.rendering import(
    function_to_latex
)
from algebraic.mercier_criterion.mercier_integrals import (
    drhat_squared_dr, expected_rhat_squared
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
    gs = grad_shafranov(coords, metric, j)

    state = State(
        coords, shaping, derivatives,
        metric, gs, j
    )

    state.shaping = gs_simplify_lutjens(
        state.gs, state.coords, state.shaping
    )

    r_constraint = drhat_squared_dr(state.jacobian, state.coords)

    r_constraint_true = -sp.diff(
        state.coords.r/2 * expected_rhat_squared(state.coords, state.shaping),
        state.coords.r
    )
    r_constraint_true = sp.simplify(r_constraint_true)
    r_constraint_true = r_constraint_true.collect(state.coords.ellip)

    assert sp.simplify(r_constraint-r_constraint_true)==0

    
    