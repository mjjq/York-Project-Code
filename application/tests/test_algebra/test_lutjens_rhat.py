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
    Test lutjens thesis eq 5.9 derivation
    """
    from algebraic.mercier_criterion.expand_mercier import gs_simplify_lutjens
    from algebraic.mercier_criterion.datatypes import State
    state_filename = "expand_mercier_state.pkl"
    
    try:
        state = State.from_disk(
            state_filename
        )
    except FileNotFoundError:
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

    rhat_squared_r = drhat_squared_dr(state.jacobian, state.coords)
    expected_rhat_squared_r = sp.expand(-state.coords.r/2 * sp.diff(
        expected_rhat_squared(state.coords, state.shaping),
        state.coords.r
    )).collect(state.coords.ellip)

    function_to_latex([
        rhat_squared_r, expected_rhat_squared_r,
        sp.simplify(rhat_squared_r-expected_rhat_squared_r)
        ])

    assert sp.simplify(rhat_squared_r-expected_rhat_squared_r)==0

    
    