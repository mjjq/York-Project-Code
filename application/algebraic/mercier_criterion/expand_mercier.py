import sympy as sp
from argparse import ArgumentParser
from copy import deepcopy

from algebraic.mercier_criterion.datatypes import (
    ToroidalCoordinates, 
    EPFLShaping, 
    LutjensShaping,
    Derivatives,
    MetricTensor,
    GradShafVariables,
    State
)
from algebraic.mercier_criterion.rendering import function_to_latex, summarise_to_latex
from algebraic.mercier_criterion.identities import (
    trig_power_to_sum, 
    sin2n_to_cos2n, 
    collect_cos_coefs,
    get_cos_coefs,
    expand_sin_terms,
    expand_cos_terms,
    remove_common_factors,
    remove_derivative_order,
    scale_r_derivatives,
    fourier_coefficients,
    fourier_expression
)


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
        R, Z, R0, r, omega, epsilon, ellip
    )
    shaping = EPFLShaping(
        D, S, m, N, P
    )
    return coords, shaping

def RZ_lutjens(include_ellip: bool = True) -> ToroidalCoordinates:
    r, omega, epsilon, ellip, R0 = sp.symbols(
        "r omega epsilon e R0"
    )
    D = sp.Function("Delta")
    E = sp.Function("E")
    T = sp.Function("T")
    P = sp.Function("P")

    ellip_tag = 1
    if not include_ellip:
        ellip_tag = 0

    R = (
        R0 - epsilon*(r-ellip_tag*ellip*E(r)) * sp.cos(omega) - epsilon**2 * D(r)  + 
        epsilon**2 * T(r) * sp.cos(2*omega) + 
        epsilon**3 * P(r) * sp.cos(omega)
    )
    Z = (
        epsilon * (r+ellip_tag*ellip*E(r)) * sp.sin(omega) +
        epsilon**2 * T(r) * sp.sin(2*omega) -
        epsilon**3 * P(r) * sp.sin(omega)
    )

    coords = ToroidalCoordinates(
        R, Z, R0, r, omega, epsilon, ellip
    )
    shaping = LutjensShaping(
        D, E, T, P
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


def order_truncation(cosine_harmonic: int) -> tuple[int, int]:
    """
    The epsilon and ellipticity ordering for a given cosine harmonic.

    See just below A.4 in Lutjens 1992:

    'Equations for the
    Shafranov shift and the elliptic and the triangular
    deformation of the flux surfaces are obtained from the
    cosw component at O ( epsilon ), the cos2w component at O(e)
    and the cos3w component at O ( epsilon ) ,
    respectively'
    """
    if cosine_harmonic==0:
        order_epsilon = 1
        order_ellip=1
    elif cosine_harmonic==1:
        order_epsilon=2
        order_ellip=2
    elif cosine_harmonic==2:
        order_epsilon = 1
        order_ellip = 2
    elif cosine_harmonic==3:
        order_epsilon = 2
        order_ellip = 1
    else:
        raise ValueError("Harmonic not supported")
    
    return order_epsilon, order_ellip


def grad_shafranov(coords: ToroidalCoordinates, 
                   tens: MetricTensor,
                   jac: sp.Expr) -> GradShafVariables:
    """
    Expand grad shafranov equation expression in Lutjens 1992
    Eq A.3, with both sides multiplied by R^2

    Note: inv_jac is J^(-1), which can be calculated using 
    calc_inv_jacobian()
    """
    f = sp.Function('f')
    g = sp.Function('g')
    a = sp.Function('a')
    p = sp.Function('p')
    B0 = sp.Symbol('B0')
    r = coords.r

    tens_gww = simplify_full(tens.g_ww, coords, 4)
    diffand = f(r)*tens_gww/jac
    diffand = sp.series(diffand, coords.epsilon, 0, 2).removeO()
    diffand = sp.series(diffand, coords.ellip, 0, 2).removeO()
    diff_r = sp.diff(diffand, r)
    print("diff_r calculated")

    tens_grw = simplify_full(tens.g_rw, coords, 4)
    diffand_omega = tens_grw/jac
    diffand_omega = sp.series(diffand_omega, coords.epsilon, 0, 2).removeO()
    diffand_omega = sp.series(diffand_omega, coords.ellip, 0, 2).removeO()
    diff_omega = f(r)*sp.diff(diffand_omega, coords.omega)
    print("diff_omega calculated")

    difference = diff_r-diff_omega
    first_term = f(r)*difference*coords.epsilon

    print("Multiplied")

    # p' = 1/eps * dp/dr
    p_term = (
        sp.diff(p(r), r) / coords.epsilon / 
        (coords.R0**2 * B0**2) * jac
    )
    p_term = truncate_expr(p_term, coords, 3)

    # In Lutjens 1992, g(r)=1+O(eps^2)
    # Let g(r)=1+a(r), s.t. a(r)~O(eps^2)
    # g'(r) = a'(r)
    # g'(r) = 1/eps * dg/dr = 1/eps * da/dr ~ O(eps)
    # gg'(r) = (1+a(r))*a'(r) = a'(r) + a(r)*a'(r)
    # => gg'(r) = a'(r) + O(eps^3) = g'(r) + O(eps^3)
    # Therefore, to O(eps), gg'(r) = g'(r)
    g_term = (
        -sp.diff(p(r), r)/B0**2 - 
        (f(r)*sp.diff(f(r),r) + f(r)**2 / r)
    )
    g_term = g_term /coords.R**2 /coords.epsilon
    g_term = g_term * jac
    g_term = truncate_expr(g_term, coords, 3, 2)

    total = first_term + g_term + p_term

    cos_coefs, sin_coefs = fourier_coefficients(
        total, coords.omega
    )

    cos_0 = cos_coefs[0]
    cos_1 = cos_coefs[1]
    cos_2 = cos_coefs[2]
    cos_3 = cos_coefs[3]

    cos_1 = sp.collect(cos_1, coords.ellip)
    cos_3 = sp.collect(cos_3, coords.ellip)

    return GradShafVariables(
        cos_0, cos_1, cos_2, cos_3, f, g, p, B0
    )

def gs_simplify_lutjens(gs: GradShafVariables, 
                        coords: ToroidalCoordinates, 
                        shaping: LutjensShaping) -> LutjensShaping:
    """
    Simplify the Lutjens GS equations by eliminating extra d2/dr2 terms.

    These extra d2/dr2 are O(e) in A.5 and A.7. Because of this, we can 
    substitute in the O(e^0) values of d2Delta/dr2, d2T/dr2, d2E/dr2
    as the O(e) components become O(e^2) and are thus eliminated.
    """
    r = coords.r

    # Get d2Delta/dr2, d2E/dr2 and d2T/dr2 to O(e)
    Delta = shaping.D
    gs_1omega_order_1 = sp.series(gs.gs_1omega, coords.ellip, 0, 1).removeO()
    delta_rr = sp.solve(
        gs_1omega_order_1, sp.diff(Delta(r), r, 2)
    )[0]

    E = shaping.E
    E_rr = sp.solve(
        gs.gs_2omega, sp.diff(E(r), r, 2)
    )[0]

    T = shaping.T
    gs_3omega_order_1 = sp.series(gs.gs_3omega, coords.ellip, 0, 1).removeO()
    T_rr = sp.solve(
        gs_3omega_order_1, sp.diff(T(r), r, 2)
    )[0]

    shaping_copy = deepcopy(shaping)
    # Now begin substituting E_rr and T_rr into the cos(omega)
    # harmonic
    subs_1omega = {
        sp.diff(E(r), r, 2): E_rr,
        sp.diff(T(r), r, 2): T_rr
    }
    

    shaping_copy.D_rr = sp.solve(
        gs.gs_1omega.subs(subs_1omega),
        sp.diff(Delta(r), r, 2)
    )[0]
    shaping_copy.D_rr = sp.collect(
        sp.simplify(
            sp.series(
                shaping_copy.D_rr, 
                coords.ellip, 0, 2
            ).removeO()
        ), 
        (coords.ellip, shaping.E(r))
    )

    # Now do similar for the cos(3omega) harmonic
    subs_3omega = {
        sp.diff(E(r), r, 2): E_rr,
        sp.diff(Delta(r), r, 2): delta_rr
    }

    shaping_copy.T_rr = sp.solve(
        gs.gs_3omega.subs(subs_3omega),
        sp.diff(T(r), r, 2)
    )[0]
    shaping_copy.T_rr = sp.collect(
        sp.simplify(
            sp.series(
                shaping_copy.T_rr, 
                coords.ellip, 0, 2
            ).removeO()
        ), 
        coords.ellip
    )

    shaping_copy.E_rr = E_rr

    return shaping_copy

def truncate_expr_fast(f: sp.Expr,
                       coords: ToroidalCoordinates,
                       order_epsilon: int = 3,
                       order_ellip: int = 2) -> sp.Expr:
    series = f

    if coords.epsilon in f.free_symbols:
        series = (series + sp.O(coords.omega**(order_epsilon-1))).removeO()

    if coords.ellip in f.free_symbols:
                series = (series + sp.O(coords.omega**(order_ellip-1))).removeO()

    return series

def truncate_expr(f: sp.Expr,
                  coords: ToroidalCoordinates,
                  order_epsilon: int = 3,
                  order_ellip: int = 2) -> sp.Expr:
    series = f

    if coords.ellip in f.free_symbols:
            series = sp.series(series, coords.ellip, 0, order_ellip).removeO()

    if coords.epsilon in f.free_symbols:
        series =  sp.series(series, coords.epsilon, 0, order_epsilon).removeO()


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
    parser = ArgumentParser()
    parser.add_argument(
        "-gs", "--gs-funcs-filename",
        type=str,
        help="Path to pre-calculated grad shafranov terms",
        default=None
    )
    args = parser.parse_args()

    

    if args.gs_funcs_filename:
        state = State.from_disk(
            args.gs_funcs_filename
        )
    else:
        coords, shaping = RZ_lutjens()
        derivatives = calc_derivatives(coords)
        metric = calc_metric_tensor(derivatives)
        j = calc_jacobian(coords, derivatives)
        gs = grad_shafranov(coords, metric, j)

        state = State(
            coords, shaping, derivatives,
            metric, gs, j
        )
        state.to_disk(
            "expand_mercier_state.pkl"
        )

    state.shaping = gs_simplify_lutjens(
        state.gs, state.coords, state.shaping
    )

    function_to_latex(
        state.gs.gs_equil, "cos_0.tex"
    )
    function_to_latex(
        state.shaping.D_rr, "cos_1.tex"
    )
    function_to_latex(
        state.shaping.E_rr, "cos_2.tex"
    )
    function_to_latex(
        state.shaping.T_rr, "cos_3.tex"
    )