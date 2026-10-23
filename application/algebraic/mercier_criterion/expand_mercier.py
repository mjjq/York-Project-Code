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

def calc_inv_jacobian(jac: sp.Function, order_epsilon: int = 2) -> sp.Function:
    """
    Calculate truncated inverse jacobian collected by 
    cosine terms (e.g. 1/J = A*cos(omega) + B*cos(2*omega)...)
    """
    # Need to perform series truncations here to avoid 
    # ungodly computation times
    # 1/J ~ O(1/eps**2), so multiply by eps**2, perform
    # series expansion, then divide by eps**2 again
    inv_jac = (1/coords.epsilon**2)*(
            simplify_full(coords.epsilon**2/jac, coords, order_epsilon)
    )
    #function_to_latex(coords.R0**2 * coords.epsilon**2 * coords.r**3 * sp.expand(inv_jac)); return

    inv_jac = sin2n_to_cos2n(inv_jac, coords.omega)
    #inv_jac = trig_power_to_sum(inv_jac, coords.omega)
    #inv_jac = collect_cos_coefs(inv_jac)

    return inv_jac


def grad_shafranov(coords: ToroidalCoordinates, 
                   tens: MetricTensor, 
                   inv_jac: sp.Function) -> GradShafVariables:
    """
    Expand grad shafranov equation expression in Lutjens 1992
    Eq A.3

    Note: inv_jac is J^(-1), which can be calculated using 
    calc_inv_jacobian()
    """
    f = sp.Function('f')
    g = sp.Function('g')
    p = sp.Function('p')
    B0 = sp.Symbol('B0')
    r = coords.r

    
    # RHS of A.3==0 so just multiply inv_jac by eps**2
    # to remove 1/eps**2 terms
    inv_jac = coords.epsilon**2 * inv_jac

    tens_gww = simplify_full(tens.g_ww, coords, 3)
    diffand = f(r)*tens_gww*inv_jac
    diff_r = sp.diff(diffand, r)

    tens_grw = simplify_full(tens.g_rw, coords, 3)
    diffand_omega = f(r)*tens_grw*inv_jac
    diff_omega = sp.diff(diffand_omega, coords.omega)

    difference = diff_r-diff_omega
    first_term = f(r)*inv_jac*difference/coords.epsilon**2
    print("Multiplied")
    first_term = truncate_expr(
        first_term, coords, order_epsilon=1, order_ellip=2
    )
    print("Truncated")
    first_term = sin2n_to_cos2n(first_term, coords.omega)
    print("sin2n to cos2n done")
    first_term = trig_power_to_sum(first_term, coords.omega)
    print("expanded cosine powers")
    first_term = collect_cos_coefs(first_term)
    print("Collected cos coefs")
    function_to_latex(first_term); return


    return

    p_term = sp.diff(p(r), r)/coords.epsilon / (coords.R0 * B0)

    g_term = g(r) * sp.diff(g(r),r)/coords.epsilon / coords.R**2

    gs = (f(r)/jac) * (diff_r - diff_omega) + p_term + g_term

    return GradShafVariables(
        gs, f, g, p, B0
    )

def truncate_expr_fast(f: sp.Expr,
                       coords: ToroidalCoordinates,
                       order_epsilon: int = 3,
                       order_ellip: int = 2) -> sp.Expr:
    series = f

    if coords.epsilon in f.free_symbols:
        series = (series + sp.O(coords.omega**order_epsilon)).removeO()

    if coords.ellip in f.free_symbols:
                series = (series + sp.O(coords.omega**order_ellip)).removeO()

    return series

def truncate_expr(f: sp.Expr,
                  coords: ToroidalCoordinates,
                  order_epsilon: int = 3,
                  order_ellip: int = 2) -> sp.Expr:
    series = f

    if coords.epsilon in f.free_symbols:
        series =  sp.series(series, coords.epsilon, 0, order_epsilon).removeO()

    if coords.ellip in f.free_symbols:
        series = sp.series(series, coords.ellip, 0, order_ellip).removeO()

    return series

def simplify(f: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.trigsimp(f))


def simplify_full(f: sp.Expr, 
                  coords: ToroidalCoordinates, 
                  order_epsilon: int = 3,
                  order_ellip: int = 2) -> sp.Expr:
    """
    Simplify equation to O(eps**2 * ellip)
    If ellip not specified, just simplify to O(eps**2)
    """
    series = truncate_expr(f, coords, order_epsilon, order_ellip)

    return simplify(series)

if __name__=='__main__':
    coords, shaping = RZ_lutjens()
    
    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)
    inv_jac = calc_inv_jacobian(j)

    #inv_jac = trig_power_to_sum(inv_jac, coords.omega)
    #inv_jac = collect_cos_coefs(inv_jac)    
    #function_to_latex([inv_jac])

    gs = grad_shafranov(coords, metric, inv_jac)

    #function_to_latex(gs.gs)
    #print(simplify_full(gs.gs, coords))


    #summarise_to_latex(coords)