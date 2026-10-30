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

    # Equation 5.9 in lutjens thesis is simplified by multiplying
    # both sides by a factor of 2/r, so recover the unsimplified
    # version before differentiating.
    # Also, lutjens' jacobian definition is the negative of ours,
    # so need to negate the result as well.
    r_constraint_true = -sp.diff(
        state.coords.r/2 * expected_rhat_squared(state.coords, state.shaping),
        state.coords.r
    )
    r_constraint_true = sp.simplify(r_constraint_true)
    r_constraint_true = r_constraint_true.collect(state.coords.ellip)

    assert sp.simplify(r_constraint-r_constraint_true)==0

    

def test_safety_factor():
    """
    Test lutjens thesis eq 5.10 derivation, safety factor.
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

    # state.shaping = gs_simplify_lutjens(
    #     state.gs, state.coords, state.shaping
    # )

    q = safety_factor(
        state.jacobian, state.coords, state.gs.B0
    )

    a = sp.Function('a')
    g = 1+state.coords.epsilon**2 * a(state.coords.r)
    G = state.coords.R0 * state.gs.B0 * g

    f = sp.Function('f')

    r_constraint = drhat_squared_dr(
        state.jacobian, state.coords
    )
    # Take the r_constraint, multiply by some common factors
    # and subtract from the safety factor (since r_constraint==0)
    r_constraint = sp.series(
        r_constraint, state.coords.ellip, 0, 1
    ).removeO()
    r_constraint = (
        state.coords.epsilon**2 / f(state.coords.r) / state.coords.R0 *
        r_constraint
    ) 
    r_constraint = sp.collect(sp.expand(r_constraint), state.coords.epsilon)
    q = sp.simplify(q - r_constraint)
    
    q_true = (
        G * state.coords.r / state.coords.R0/ 
        (state.coords.R0 * state.gs.B0 * f(state.coords.r))
    )

    assert sp.simplify(q-q_true)==0
