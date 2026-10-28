from algebraic.mercier_criterion.expand_mercier import (
    RZ_lutjens, calc_jacobian, calc_inv_jacobian,
    calc_metric_tensor, calc_derivatives,
    grad_shafranov
)
from algebraic.mercier_criterion.rendering import(
    function_to_latex
)
import sympy as sp
from sympy import Derivative

def test_lutjens_zero_ellip():
    """
    Test that we can recover equations A.5 and A.7 in Lutjens
    1992 for the ellip=0 case
    """
    coords, shaping = RZ_lutjens(include_ellip=False)

    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)
    inv_jac = calc_inv_jacobian(j, coords)

    gs = grad_shafranov(coords, metric, j, inv_jac)

    r = coords.r
    R0 = coords.R0
    Delta = coords.D
    f = gs.f
    B0 = gs.B0
    p = gs.p

    # First test the cos(omega) component
    cos_1 = gs.gs_1omega
    # Multiply all terms of A.5 by factor rf**2
    # Not important since the expr is equal to zero
    # anyway
    cos_1_true = r*f(r)**2 * (
        Derivative(Delta(r), (r, 2)) + 
        Derivative(Delta(r), r)*(
            2*Derivative(f(r), r)/f(r) + 
            1/r
        ) -
        1/R0 + 2*r*Derivative(p(r), r)/(B0**2*R0*f(r)**2)
    )
    assert sp.simplify(cos_1-cos_1_true)==0

    # Now test the cos(3omega) component
    T = shaping.T
    cos_3 = gs.gs_3omega
    # Again multiply both of A.7 sides by rf**2
    cos_3_true = -r*f(r) * (
        Derivative(T(r), (r, 2)) + 
        Derivative(T(r), r)*(
            2*Derivative(f(r), r)/f(r) + 
            1/r
        ) -
        8*T(r)/r**2
    )
    assert sp.simplify(cos_3-cos_3_true)==0


def test_lutjens_cos2():
    """
    Test that we can recover equation A.6 (cos(2omega)) in 
    Lutjens 1992 for the ellip=0 case
    """
    coords, shaping = RZ_lutjens()

    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)
    inv_jac = calc_inv_jacobian(j, coords)

    gs = grad_shafranov(coords, metric, j, inv_jac)

    r = coords.r
    R0 = coords.R0
    Delta = coords.D
    f = gs.f
    B0 = gs.B0
    p = gs.p
    E = shaping.E

    # First test the cos(omega) component
    cos_2 = gs.gs_2omega
    # Multiply all terms of A.6 by factor rf**2
    # Not important since the expr is equal to zero
    # anyway
    cos_2_true = -r*f(r) * (
        Derivative(E(r), (r, 2)) + 
        Derivative(E(r), r)*(
            2*Derivative(f(r), r)/f(r) + 
            1/r
        ) -
        3*E(r)/r**2
    )
    assert sp.simplify(cos_2-cos_2_true)==0