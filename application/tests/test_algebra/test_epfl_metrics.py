from algebraic.mercier_criterion.expand_mercier import RZ_epfl, calc_jacobian, calc_metric_tensor, calc_derivatives
import sympy as sp

def test_g_ww_eps2():
    """
    Test Jon's g_ww calculation to O(epsilon^2)
    """
    coords, shaping = RZ_epfl()

    derivs = calc_derivatives(coords)
    tens = calc_metric_tensor(derivs)

    trunc_g_ww = sp.series(tens.g_ww, coords.epsilon, 0, 4).removeO()

    S=shaping.S
    m=shaping.m
    N=shaping.N
    expr_sum = sp.Sum((m-1)*S(m,coords.r) * sp.cos(m*coords.omega), (m, 2, N))

    true_g_ww = coords.epsilon**2 * coords.r**2 * (
        1 - 2*coords.epsilon/coords.r * expr_sum
    )

    assert sp.simplify(sp.trigsimp(sp.simplify(trunc_g_ww-true_g_ww)))==0

def test_g_rw_eps2():
    """
    Test Jon's g_rw calculation to O(epsilon^2)
    """
    coords, shaping = RZ_epfl()

    derivs = calc_derivatives(coords)
    tens = calc_metric_tensor(derivs)

    trunc_g_rw = sp.series(tens.g_rw, coords.epsilon, 0, 4).removeO()

    S = shaping.S
    m = shaping.m
    N = shaping.N
    expr_sum = sp.Sum(
        ((m-1)*S(m,coords.r)/coords.r + sp.diff(S(m,coords.r), coords.r))* sp.sin(m*coords.omega), 
        (m, 2, N)
    )

    true_g_rw = coords.epsilon**3 * coords.r * (
        sp.diff(coords.D(coords.r), coords.r)*sp.sin(coords.omega) - expr_sum
    )

    assert sp.simplify(sp.trigsimp(sp.simplify(trunc_g_rw-true_g_rw)))==0

def test_jacobian_eps2():
    """
    Test Jon's Jacobian calculation to O(eps^2)
    """
    coords, shaping = RZ_epfl()

    derivs = calc_derivatives(coords)

    jac = calc_jacobian(coords, derivs)

    trunc_jac = sp.simplify(sp.series(jac, coords.epsilon, 0, 4).removeO())

    print(trunc_jac)

    S, m, N = shaping.S, shaping.m, shaping.N
    expr_sum = sp.Sum(
        (-(m-1)*S(m,coords.r)/coords.r + sp.diff(S(m,coords.r), coords.r))* sp.cos(m*coords.omega), 
        (m, 2, N)
    )

    true_jac = sp.simplify(coords.epsilon**2 * coords.r * coords.R0 * (
        1 + 
        coords.epsilon * (coords.r/coords.R0 - sp.diff(coords.D(coords.r), coords.r)) * sp.cos(coords.omega) +
        coords.epsilon * expr_sum
    ))

    assert sp.simplify(trunc_jac-true_jac)==0