import sympy as sp

from algebraic.mercier_criterion.datatypes import (
    ToroidalCoordinates, 
    EPFLShaping, 
    LutjensShaping,
    Derivatives,
    MetricTensor,
    GradShafVariables
)
from algebraic.mercier_criterion.rendering import function_to_latex, summarise_to_latex
from algebraic.mercier_criterion.identities import trig_power_to_sum, sin2n_to_cos2n, collect_cos_coefs


def RZ_epfl() -> tuple[ToroidalCoordinates, EPFLShaping]:
    r, omega, epsilon, ellip, R0 = sp.symbols(
        "r omega epsilon ellip R0"
    )
    D = sp.Function("Delta")
    P = sp.Function("P")

    # For fourier summation
    m, N = sp.symbols("m N")
    # Fourier coefficients
    S = sp.Function("S")
    cos_expr = sp.Sum(S(m,r) * sp.cos((m-1)*omega), (m, 2, N))
    sin_expr = sp.Sum(S(m,r) * sp.sin((m-1)*omega), (m, 2, N))

    R = (
        R0 + epsilon * r * sp.cos(omega) - epsilon**2 * D(r) + 
        epsilon**2 * cos_expr + 
        epsilon**3 * P(r) * sp.cos(omega)
    )
    Z = (
        epsilon * r * sp.sin(omega) -
        epsilon**2 * sin_expr +
        epsilon**3 * P(r) * sp.sin(omega)
    )

    coords = ToroidalCoordinates(
        R, Z, D, P, R0, r, omega, epsilon, ellip
    )
    shaping = EPFLShaping(
        S, m, N
    )
    return coords, shaping

def RZ_lutjens() -> ToroidalCoordinates:
    r, omega, epsilon, ellip, R0 = sp.symbols(
        "r omega epsilon e R0"
    )
    D = sp.Function("Delta")
    E = sp.Function("E")
    T = sp.Function("T")
    P = sp.Function("P")

    R = (
        R0 - epsilon*(r-ellip*E(r)) * sp.cos(omega) - epsilon**2 * D(r)  + 
        epsilon**2 * T(r) * sp.cos(2*omega) + 
        epsilon**3 * P(r) * sp.cos(omega)
    )
    Z = (
        epsilon * (r+ellip*E(r)) * sp.sin(omega) +
        epsilon**2 * T(r) * sp.sin(2*omega) -
        epsilon**3 * P(r) * sp.sin(omega)
    )

    coords = ToroidalCoordinates(
        R, Z, D, P, R0, r, omega, epsilon, ellip
    )
    shaping = LutjensShaping(
        E, T
    )

    return coords, shaping


def calc_derivatives(coords: ToroidalCoordinates) -> Derivatives:
    return Derivatives(
        sp.diff(coords.R, coords.r),
        sp.diff(coords.R, coords.omega),
        sp.diff(coords.Z, coords.r),
        sp.diff(coords.Z, coords.omega)
    )


def calc_metric_tensor(derivatives: Derivatives) -> MetricTensor:
    g_rr = derivatives.R_r**2 + derivatives.Z_r**2
    g_rw = (
        derivatives.R_r*derivatives.R_omega + 
        derivatives.Z_r*derivatives.Z_omega
    )
    g_ww = derivatives.R_omega**2 + derivatives.Z_omega**2

    return MetricTensor(
        g_rr, g_rw, g_ww
    )

def calc_jacobian(coords: ToroidalCoordinates, 
                  derivatives: Derivatives) -> sp.Function:
    return coords.R * (
        derivatives.R_r * derivatives.Z_omega -
        derivatives.R_omega * derivatives.Z_r
    )


def grad_shafranov(coords: ToroidalCoordinates, 
                   tens: MetricTensor, 
                   jac: sp.Function) -> GradShafVariables:
    """
    Expand grad shafranov equation expression in Lutjens 1992
    Eq A.3
    """
    f = sp.Function('f')
    g = sp.Function('g')
    p = sp.Function('p')
    B0 = sp.Symbol('B0')
    r = coords.r

    # Need to perform series truncations here to avoid 
    # ungodly computation times
    # 1/J ~ O(1/eps**2), so multiply by eps**2, perform
    # series expansion, then divide by eps**2 again
    inv_jac = (1/coords.epsilon**2)*(
            simplify_full(coords.epsilon**2/jac, coords, 2)
    )
    #function_to_latex(coords.R0**2 * coords.epsilon**2 * coords.r**3 * sp.expand(inv_jac)); return

    #inv_jac = trig_power_to_sum(inv_jac, coords.omega)
    # cos_terms = sorted(
    #     sp.expand(inv_jac).atoms(sp.cos),
    #     key=sp.default_sort_key
    # )
    #result = sp.collect(inv_jac, cos_terms)
    #result = sp.expand(inv_jac).coeff(sp.cos(2*coords.omega))
    inv_jac = sin2n_to_cos2n(inv_jac, coords.omega)
    inv_jac = trig_power_to_sum(inv_jac, coords.omega)
    inv_jac = collect_cos_coefs(inv_jac)

    function_to_latex(inv_jac); return

    # Need g_ww to go to O(eps**4) since 
    # 1/J ~ 1/eps**2 to lowest order
    tens_gww = simplify_full(tens.g_ww, coords, 4)
    prod = tens_gww*inv_jac
    #function_to_latex(prod); return

    diffand = f(r)*simplify_full(tens_gww*inv_jac, coords)

    #function_to_latex(diffand); return

    diff_r = sp.diff(diffand, r)

    #function_to_latex(diff_r); return

    tens_grw = simplify_full(tens.g_rw, coords)
    diffand_omega = f(r)*simplify_full(tens_grw*inv_jac, coords)
    diff_omega = simplify_full(sp.diff(diffand_omega, coords.omega), coords)
    
    difference = simplify_full(diff_r - diff_omega, coords)


    first_term = simplify_full(f(r)*inv_jac*difference, coords)
    function_to_latex(first_term)

    return

    p_term = sp.diff(p(r), r)/coords.epsilon / (coords.R0 * B0)

    g_term = g(r) * sp.diff(g(r),r)/coords.epsilon / coords.R**2

    gs = (f(r)/jac) * (diff_r - diff_omega) + p_term + g_term

    return GradShafVariables(
        gs, f, g, p, B0
    )


def simplify_full(f: sp.Function, 
                  coords: ToroidalCoordinates, 
                  order_epsilon: int = 3,
                  order_ellip: int = 2) -> sp.Function:
    """
    Simplify equation to O(eps**2 * ellip)
    If ellip not specified, just simplify to O(eps**2)
    """
    series = f

    if coords.epsilon in f.free_symbols:
        series =  sp.series(series, coords.epsilon, 0, order_epsilon).removeO()

    if coords.ellip in f.free_symbols:
        series = sp.series(series, coords.ellip, 0, order_ellip).removeO()

    return sp.simplify(sp.trigsimp(series))

if __name__=='__main__':
    coords, shaping = RZ_lutjens()
    
    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)

    gs = grad_shafranov(coords, metric, j)

    #function_to_latex(gs.gs)
    #print(simplify_full(gs.gs, coords))


    #summarise_to_latex(coords)