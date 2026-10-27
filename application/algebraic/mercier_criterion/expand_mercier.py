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
from algebraic.mercier_criterion.identities import (
    trig_power_to_sum, 
    sin2n_to_cos2n, 
    collect_cos_coefs,
    get_cos_coefs,
    expand_sin_terms,
    expand_cos_terms,
    remove_common_factors,
    remove_derivative_order
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
        R0 - epsilon*(r-0*E(r)) * sp.cos(omega) - epsilon**2 * D(r)  + 
        epsilon**2 * T(r) * sp.cos(2*omega) + 
        epsilon**3 * P(r) * sp.cos(omega)
    )
    Z = (
        epsilon * (r+0*E(r)) * sp.sin(omega) +
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

def calc_inv_jacobian(jac: sp.Function, coords, order_epsilon: int = 2) -> sp.Function:
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

def gs_simplify(first_term: sp.Expr, 
                coords: ToroidalCoordinates,
                cosine_harmonic: int = 1) -> list[sp.Expr]:
    # order_epsilon, order_ellip = order_truncation(cosine_harmonic)
    # first_term = truncate_expr(
    #     first_term, coords, 
    #     order_epsilon=order_epsilon, 
    #     order_ellip=order_ellip
    # )
    first_term = expand_sin_terms(first_term, coords.omega)
    print("Expanded sin(mx) terms")
    first_term = expand_cos_terms(first_term, coords.omega)
    print("Expanded cos(mx) terms")
    first_term = sin2n_to_cos2n(first_term, coords.omega)
    print("sin2n to cos2n done")
    first_term = trig_power_to_sum(first_term, coords.omega)
    print("expanded cosine powers")
    first_term = collect_cos_coefs(first_term)
    print("Collected cos coefs")
    return first_term

def extract_cosine_equation(terms: list[sp.Expr],
                            coords: ToroidalCoordinates,
                            cosine_harmonic: int) -> sp.Expr:
    """
    Extract cosine harmonic equation from a list of
    terms containing a sum of harmonics

    Any repeated terms are automatically excluded
    using set()
    """
    cosine_terms = set()
    for term in terms:
        coef = get_cos_coefs(
            term, coords.omega, [cosine_harmonic]
        )[0]
        cosine_terms.add(coef)

    sum_of_terms = sp.Add(*cosine_terms)
    return sp.expand(remove_common_factors(sum_of_terms))


def grad_shafranov(coords: ToroidalCoordinates, 
                   tens: MetricTensor,
                   jac: sp.Expr, 
                   inv_jac: sp.Expr) -> GradShafVariables:
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
    diffand = f(r)*tens_gww*inv_jac
    diffand = truncate_expr(diffand, coords, 4)
    diff_r = sp.diff(diffand, r)
    #diff_r = gs_simplify(diff_r, coords)

    tens_grw = simplify_full(tens.g_rw, coords, 4)
    diffand_omega = tens_grw*inv_jac
    diffand_omega = truncate_expr(diffand_omega, coords, 4)
    diff_omega = f(r)*sp.diff(diffand_omega, coords.omega)
    #diff_omega = gs_simplify(diff_omega, coords)

    difference = diff_r-diff_omega

    
    first_term = coords.epsilon*f(r)*difference
    first_term = truncate_expr(first_term, coords, 3)
    first_term = gs_simplify(first_term, coords)

    print("Multiplied")
    # cos_2 = get_cos_coefs(first_term, coords.omega, [2])
    # cos_2_eq = extract_cosine_equation([first_term], coords, 2)
    # function_to_latex([first_term, cos_2, cos_2_eq]); return

    # return

    # p' = 1/eps * dp/dr
    p_term = sp.diff(p(r), r) / coords.epsilon / (coords.R0**2 * B0**2) * jac
    p_term = truncate_expr(p_term, coords, 3)
    p_term = gs_simplify(p_term, coords, 1)

    # In Lutjens 1992, g(r)=1+O(eps^2)
    # Let g(r)=1+a(r), s.t. a(r)~O(eps^2)
    # g'(r) = a'(r)
    # g'(r) = 1/eps * dg/dr = 1/eps * da/dr ~ O(eps)
    # gg'(r) = (1+a(r))*a'(r) = a'(r) + a(r)*a'(r)
    # => gg'(r) = a'(r) + O(eps^3) = g'(r) + O(eps^3)
    # Therefore, to O(eps), gg'(r) = g'(r)
    #
    # Now, 1/R^2 ~ 1/R0^2 + O(eps^2)
    # So gg'/R^2 = [g'(r)+O(eps^3)][1/R0^2 + O(eps^2)] = g'(r)/R0^2 + O(eps^2)
    # Therefore, to order eps, gg'/R^2 ~ g'/R0^2
    # No need to use sympy to auto expand gg'/R^2
    # inv_R_squared = 
    inv_R_squared = truncate_expr(1/coords.R**2, coords, 3, 2)
    #inv_R_squared = gs_simplify(inv_R_squared, coords, 1)
    #function_to_latex(inv_R_squared); return
    g_term = -sp.diff(p(r), r)/B0**2 - (f(r)*sp.diff(f(r),r) + f(r)**2 / r)
    g_term = g_term * inv_R_squared/coords.epsilon
    g_term = truncate_expr(g_term, coords, 3, 2)
    g_term = g_term * jac
    g_term = truncate_expr(g_term, coords, 3, 2)

    g_term = gs_simplify(g_term, coords, 1)
    #function_to_latex([p_term, g_term]); return
    
    # g_term_0_2 = gs_simplify(g_term, coords, 2)
    # g_term_0_1_3 = gs_simplify(g_term, coords, 3)
    #function_to_latex([first_term, p_term, g_term]); return

    terms = [
        first_term, g_term, p_term
    ]
    #function_to_latex(terms); return

    cos_0 = extract_cosine_equation(terms, coords, 0)
    cos_1 = extract_cosine_equation(terms, coords, 1)
    cos_2 = extract_cosine_equation(terms, coords, 2)
    cos_3 = extract_cosine_equation(terms, coords, 3)

    # Remove derivatives higher order than 2, e.g.
    # ~ d2E/dr2 * dDelta/dr
    # This is because the equations later convert back
    # to prime notation, and ' = 1/eps d/dr.
    # So converting from d/dr to prime introduces a factor
    # epsilon. The derivatives with order higher than 2
    # are of order epsilon^3 when converted to primes. 
    # Other terms are at least O(epsilon), so we wish
    # to only preserve terms which are O(epsilon^2)
    # cos_1 = remove_derivative_order(cos_1)
    # cos_1 = sp.collect(cos_1, coords.ellip)
    # cos_3 = remove_derivative_order(cos_3)
    # cos_3 = sp.collect(cos_3, coords.ellip)

    # cos_0 = truncate_expr(cos_0, coords, *order_truncation(0))
    # cos_1 = truncate_expr(cos_1, coords, *order_truncation(1))
    # cos_2 = truncate_expr(cos_2, coords, *order_truncation(2))
    # cos_3 = truncate_expr(cos_3, coords, *order_truncation(3))

    function_to_latex(
        cos_0, "cos_0.tex"
    )
    function_to_latex(
        cos_1, "cos_1.tex"
    )
    function_to_latex(
        cos_2, "cos_2.tex"
    )
    function_to_latex(
        cos_3, "cos_3.tex"
    )
    return
    #gs_equil = 

    return GradShafVariables(
        gs, f, g, p, B0
    )

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

    #function_to_latex(truncate_expr(coords.R**2, coords)); exit()
    
    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)
    inv_jac = calc_inv_jacobian(j, coords)

    #inv_jac = trig_power_to_sum(inv_jac, coords.omega)
    #inv_jac = collect_cos_coefs(inv_jac)    
    #function_to_latex([inv_jac])

    gs = grad_shafranov(coords, metric, j, inv_jac)

    #function_to_latex(gs.gs)
    #print(simplify_full(gs.gs, coords))


    #summarise_to_latex(coords)