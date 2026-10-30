from algebraic.mercier_criterion.expand_mercier import (
    RZ_lutjens, calc_jacobian,
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

    gs = grad_shafranov(coords, metric, j)

    r = coords.r
    R0 = coords.R0
    Delta = shaping.D
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

    delta_rr = sp.solve(cos_1, sp.diff(Delta(r), r, 2))[0]
    delta_rr_true = sp.solve(cos_1_true, sp.diff(Delta(r), r, 2))[0]

    assert sp.simplify(delta_rr-delta_rr_true)==0

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

    T_rr = sp.solve(cos_3, sp.diff(T(r), r, 2))[0]
    T_rr_true = sp.solve(cos_3, sp.diff(T(r),r,2))[0]
    assert sp.simplify(T_rr-T_rr_true)==0


def test_lutjens_cos2():
    """
    Test that we can recover equation A.6 (cos(2omega)) in 
    Lutjens 1992 for the ellip=0 case
    """
    coords, shaping = RZ_lutjens()

    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)

    gs = grad_shafranov(coords, metric, j)

    r = coords.r
    R0 = coords.R0
    Delta = shaping.D
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

    E_rr = sp.solve(cos_2, sp.diff(E(r), r, 2))[0]
    E_rr_true = sp.solve(cos_2_true, sp.diff(E(r), r, 2))[0]
    assert sp.simplify(E_rr - E_rr_true)==0


def test_lutjens_shaping_order_eps():
    """
    Test lutjens A.5-A.7 derivations (cos(omega)-cos(3omega))
    components
    """
    from algebraic.mercier_criterion.expand_mercier import gs_simplify_lutjens
    from algebraic.mercier_criterion.datatypes import State

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


    r = state.coords.r
    R0 = state.coords.R0
    Delta = state.shaping.D
    f = state.gs.f
    B0 = state.gs.B0
    p = state.gs.p
    e = state.coords.ellip
    E = state.shaping.E
    T = state.shaping.T

    # First test the cos(omega) component
    D_rr = state.shaping.D_rr
    # Multiply all terms of A.5 by factor rf**2
    # Not important since the expr is equal to zero
    # anyway
    D_rr_true = (
        -(2*Derivative(f(r),r)/f(r) + 1/r)*Derivative(Delta(r), r) +
        1/R0 - 2*r*Derivative(p(r),r)/(R0*B0**2*f(r)**2) +
        e * (
            E(r)*(
                3*Derivative(p(r),r)/(R0*B0**2*f(r)**2) -
                3*Derivative(Delta(r), r)/r**2 +
                12*T(r)/r**3 + 
                5*Derivative(T(r),r)/r**2 +
                2*T(r)*Derivative(f(r),r)/(r**2*f(r))
            ) +
            Derivative(E(r),r) * (
                3*r*Derivative(p(r),r)/(R0*B0**2*f(r)**2) +
                3*Derivative(Delta(r), r)*Derivative(f(r),r)/f(r) +
                10*T(r)/r**2 + 
                2*Derivative(T(r),r)/r -
                3*Derivative(T(r),r)*Derivative(f(r),r)/f(r) -
                1/R0
            )
        )
    )

    assert sp.simplify(D_rr - D_rr_true)==0

    # Now test d2E/dr2
    E_rr = state.shaping.E_rr
    E_rr_true = (
        -(2*Derivative(f(r))/f(r) + 1/r) * Derivative(E(r),r) +
        3*E(r)/r**2
    )

    assert sp.simplify(E_rr-E_rr_true)==0


    # Now test d2T/dr2
    T_rr = state.shaping.T_rr
    T_rr_true = (
        -(2*Derivative(f(r))/f(r) + 1/r) * Derivative(T(r),r) +
        8*T(r)/r**2 +
        e * (
            E(r)*(
                Derivative(p(r),r)/(R0*B0**2*f(r)**2) +
                3*Derivative(Delta(r), r)/r**2
            ) -
            Derivative(E(r),r) * (
                3*r*Derivative(p(r),r)/(R0*B0**2*f(r)**2) +
                3*Derivative(Delta(r), r)*Derivative(f(r),r)/f(r) +
                4*Derivative(Delta(r), r)/r -
                1/R0
            )
        )
    )

    assert sp.simplify(T_rr - T_rr_true)==0