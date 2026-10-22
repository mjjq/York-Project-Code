import sympy as sp
from dataclasses import dataclass

def expand_small_parameter():
    epsilon = sp.symbols('epsilon')
    f = sp.exp(epsilon)/(1+epsilon)

    print(sp.series(f, epsilon, 0, 3))

def expand_small_param_x():
    a, x, epsilon = sp.symbols('a x epsilon')
    f = sp.sqrt(1 + a*epsilon*x)

    print(sp.series(f, epsilon, 0, 3))

@dataclass
class ToroidalCoordinates:
    R: sp.Function # Major radius
    Z: sp.Function # Vertical distance

    D: sp.Function # Shafranov shift (function of r)
    P: sp.Function # P(r)

    R0: sp.Symbol # Major radius of magnetic axis
    
    # New co-ords
    r: sp.Symbol # Minor radius
    omega: sp.Symbol # Generalised angle

    # Expansion params
    epsilon: sp.Symbol # Inverse aspect ratio tag
    ellip: sp.Symbol # Ellipticity tag

@dataclass
class EPFLShaping:
    # Shaping
    S: sp.Function # Shaping coefficients (S_m(r))
    m: sp.Symbol # Shaping coef indices
    N: sp.Symbol # sum_{m=1}^N...

@dataclass
class LutjensShaping:
    E: sp.Function # ellipticity E(r)
    T: sp.Function # Triangularity T(r)

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

@dataclass
class Derivatives:
    R_r: sp.Function
    R_omega: sp.Function
    Z_r: sp.Function
    Z_omega: sp.Function

def calc_derivatives(coords: ToroidalCoordinates) -> Derivatives:
    return Derivatives(
        sp.diff(coords.R, coords.r),
        sp.diff(coords.R, coords.omega),
        sp.diff(coords.Z, coords.r),
        sp.diff(coords.Z, coords.omega)
    )

@dataclass
class MetricTensor:
    g_rr: sp.Function
    g_rw: sp.Function
    g_ww: sp.Function

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

def cos_power_to_sum(expr: sp.Function, x: sp.Symbol):
    expr = sp.expand(expr)

    replacements = {}

    for power in expr.atoms(sp.Pow):
        base, exponent = power.as_base_exp()

        if base == sp.cos(x) and exponent.is_Integer and exponent >= 2:
            n = int(exponent)

            replacements[power] = sp.expand(
                sum(
                    sp.binomial(n, k) * sp.cos((n - 2*k)*x)
                    for k in range(n + 1)
                ) / 2**n
            )

    return expr.xreplace(replacements)


@dataclass
class GradShafVariables:
    gs: sp.Function # Equation A.3. in Lutjens 1992 expanded

    f: sp.Function # dpsi/dr = R0 B0 eps * f(r)
    g: sp.Function # G = R0 B0 g(r), g(r) = 1+O(eps^2)

    p: sp.Function # pressure p(r)

    B0: sp.Symbol # On-axis toroidal field

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
    inv_jac = cos_power_to_sum(inv_jac, coords.omega)
    cos_terms = sorted(
        inv_jac.atoms(sp.cos),
        key=sp.default_sort_key
    )
    result = sp.collect(inv_jac, cos_terms)
    function_to_latex(result); return

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


def summarise_expansions(coords: ToroidalCoordinates):
    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)

    print("Co-ordinates")
    sp.pprint(sp.series(coords.R, coords.epsilon, 0, 4), use_unicode=True)
    sp.pprint(sp.series(coords.Z, coords.epsilon, 0, 4), use_unicode=True)
    
    print()
    print("Metrics")

    sp.pprint(sp.series(metric.g_rw, coords.epsilon, 0, 2), use_unicode=True)
    sp.pprint(sp.series(metric.g_rr, coords.epsilon, 0, 2), use_unicode=True)
    sp.pprint(sp.trigsimp(sp.simplify(sp.series(metric.g_ww, coords.epsilon, 0, 2))), use_unicode=True)

    print()
    print("Jacobian")
    sp.pprint(sp.simplify(sp.simplify(sp.series(j, coords.epsilon, 0, 2))), use_unicode=True)

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

def summarise_to_latex(coords: ToroidalCoordinates):
    """
    Generates latex document called formula.tex.

    Run pdflatex formula.tex to render equations to pdf.
    """
    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)

    # print("Co-ordinates")
    latex = []
    latex.append(sp.latex(sp.series(coords.R, coords.epsilon, 0, 4)))
    latex.append(sp.latex(sp.series(coords.Z, coords.epsilon, 0, 4)))
    latex.append(" ")
    latex.append(" ")
    # print()
    # print("Metrics")
    
    latex.append(sp.latex(sp.series(derivatives.R_r, coords.epsilon, 0, 3)))
    latex.append(sp.latex(sp.series(derivatives.R_omega, coords.epsilon, 0, 3)))
    latex.append(sp.latex(sp.series(derivatives.Z_r, coords.epsilon, 0, 3)))
    latex.append(sp.latex(sp.series(derivatives.Z_omega, coords.epsilon, 0, 3)))
    latex.append(" ")
    latex.append(" ")

    latex.append(sp.latex(simplify_full(metric.g_rr, coords)))
    latex.append(sp.latex(simplify_full(metric.g_rw, coords)))
    latex.append(sp.latex(simplify_full(metric.g_ww, coords)))
    latex.append(" ")
    latex.append(" ")
    latex.append(sp.latex(simplify_full(j, coords)))

    # print()
    # print("Jacobian")
    # sp.pprint(sp.simplify(sp.simplify(sp.series(j, coords.epsilon, 0, 2))), use_unicode=True)

    to_latex = "\]\[".join(latex)
    document = rf"""
    \documentclass[a3paper,landscape]{{article}}
    \usepackage[left=0.1cm, right=0.1cm]{{geometry}}
    \usepackage{{amsmath}}
    \pagestyle{{empty}}
    \begin{{document}}
    \tiny
    \[
    """+to_latex+rf"""
    \]
    \end{{document}}
    """
    print(document)

    with open("formula.tex", "w") as file:
        file.write(document)

def function_to_latex(f_list: list[sp.Function]):
    print(type(f_list))
    if isinstance(f_list, list):
        latex = [sp.latex(f) for f in f_list]
    else:
        latex = [sp.latex(f_list)]

    # print()
    # print("Jacobian")
    # sp.pprint(sp.simplify(sp.simplify(sp.series(j, coords.epsilon, 0, 2))), use_unicode=True)

    to_latex = "\]\[".join(latex)
    document = rf"""
    \documentclass[a3paper,landscape]{{article}}
    \usepackage[left=0.1cm, right=0.1cm]{{geometry}}
    \usepackage{{amsmath}}
    \pagestyle{{empty}}
    \begin{{document}}
    \tiny
    \[
    """+to_latex+rf"""
    \]
    \end{{document}}
    """
    print(document)

    with open("formula.tex", "w") as file:
        file.write(document)

if __name__=='__main__':
    coords, shaping = RZ_lutjens()
    
    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)

    gs = grad_shafranov(coords, metric, j)

    #function_to_latex(gs.gs)
    #print(simplify_full(gs.gs, coords))


    #summarise_to_latex(coords)