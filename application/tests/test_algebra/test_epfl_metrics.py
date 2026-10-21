from algebraic.mercier_criterion.expand_mercier import RZ_epfl, calc_jacobian, calc_metric_tensor, calc_derivatives
import sympy as sp

def test_g_ww_eps2():
    """
    Test Jon's g_ww calculation to O(epsilon^2)
    """
    coords = RZ_epfl()

    derivs = calc_derivatives(coords)
    tens = calc_metric_tensor(derivs)

    trunc_g_ww = sp.series(tens.g_ww, coords.epsilon, 0, 2).removeO()

    m, N = sp.symbols("m N")
    S = sp.Function("S")
    expr_sum = sp.Sum((m-1)*S(m,coords.r) * sp.cos(m*coords.omega), (m, 2, N))

    true_g_ww = coords.r**2 * (
        1 - 2*coords.epsilon/coords.r * expr_sum
    )

    assert sp.simplify(sp.trigsimp(sp.simplify(trunc_g_ww-true_g_ww)))==0