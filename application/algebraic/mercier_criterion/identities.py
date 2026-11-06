import sympy as sp
import re
from sympy.core.function import AppliedUndef

from algebraic.mercier_criterion.rendering import function_to_latex

def truncate(expr: sp.Expr, x: sp.Symbol, y: sp.Symbol, total_order: int = 2) -> sp.Expr:
    """
    Truncate expression to only retain terms up to order
    x**2, y, and xy
    """
    poly = sp.Poly(sp.expand(expr), x, y)

    return sp.Add(*[
        coeff * x**i * y**j
        for (i, j), coeff in poly.terms()
        if i + j <= total_order
    ])

def fast_truncate(expr: sp.Expr, x: sp.Symbol, order: int) -> sp.Expr:
    """
    Truncate expression to only retain terms up to order
    x**2, y, and xy
    """
    poly = sp.Poly(sp.expand(expr), x)

    return sp.Add(*[
        coeff * x**i
        for (i,), coeff in poly.terms()
        if i < order
    ])

def faster_truncate(expr: sp.Expr, x: sp.Symbol, order: int) -> sp.Expr:
    """
    Truncate expression to only retain terms up to order
    x**2, y, and xy
    """
    return sum(
        term for term in sp.expand(expr).args 
        if sp.degree(term, x) < order
    )

def integrate_by_parts(u: sp.Expr, dv: sp.Expr, x: sp.Symbol) -> sp.Expr:
    v = sp.integrate(dv, x)
    du = sp.diff(u, x)

    return u*v - sp.integrate(v*du, x)

def fast_trig_expand(expr: sp.Expr, 
                     x: sp.Symbol, 
                     order: int = 2) -> sp.Expr:
    # Taylor approximations of sin(x) and cos(x)
    replacements = {
        sp.sin(x): sp.series(sp.sin(x), x, 0, order).removeO(),
        sp.cos(x): sp.series(sp.cos(x), x, 0, order).removeO()
    }

    # Replace trig functions throughout the expression
    expr = expr.xreplace(replacements)

    # Convert to polynomial and truncate higher-order terms
    poly = sp.Poly(expr, x)

    # Keep only terms with powers lower than order
    result = sum(
        coeff * x**power[0]
        for power, coeff in poly.terms()
        if power[0] < order
    )

    return result


def fast_trig_expand_2(expr: sp.Expr,
                     dx: sp.Symbol,
                     order: int = 2) -> sp.Expr:

    replacements = {}

    # Find all sine and cosine terms containing dx
    for f in expr.atoms(sp.sin, sp.cos):
        if f.args[0].has(dx):
            replacements[f] = sp.series(
                f, dx, 0, order
            ).removeO()

    # Replace all matching terms
    result = expr.xreplace(replacements)

    return sp.collect(sp.expand(result), dx)

def fourier_coefficients(expr, x):
    """
    Extract Fourier coefficients from a trigonometric polynomial.

    Supports integer-frequency harmonics:
        sin(m*x), cos(m*x)
    and products and nonnegative integer powers of these terms.

    Returns dictionaries a, b such that:
        expr = a[0]/2 + sum(
            a[k]*cos(k*x) + b[k]*sin(k*x)
            for k >= 1
        )
    """
    z = sp.Symbol('_z', nonzero=True)
    I = sp.I

    # Replace each harmonic sin(m*x), cos(m*x) by its
    # corresponding Laurent polynomial in z = exp(I*x).
    def harmonic_rewrite(e):
        if e.func == sp.sin or e.func == sp.cos:
            arg = e.args[0]

            # Match integer multiples of x
            m = sp.simplify(arg / x)

            if m.is_Integer:
                m = int(m)

                if e.func == sp.cos:
                    return (z**m + z**(-m)) / 2
                else:
                    return (z**m - z**(-m)) / (2*I)

        return e

    expr_z = expr.replace(
        lambda e: (
            e.func in (sp.sin, sp.cos)
            and sp.simplify(e.args[0] / x).is_Integer
        ),
        harmonic_rewrite
    )

    expr_z = sp.expand(expr_z)

    # Collect the coefficients of z**k, including negative k.
    coeffs = {}

    for term in sp.Add.make_args(expr_z):
        powers = term.as_powers_dict()
        exponent = powers.get(z, sp.S.Zero)

        if not exponent.is_Integer:
            raise ValueError(
                f"Non-integer Fourier exponent encountered: {exponent}"
            )

        k = int(exponent)
        coeff = sp.simplify(term / z**k)

        coeffs[k] = coeffs.get(k, 0) + coeff

    coeffs = {
        k: sp.simplify(v)
        for k, v in coeffs.items()
        if sp.simplify(v) != 0
    }

    if not coeffs:
        return {0: sp.S.Zero}, {}

    N = max(abs(k) for k in coeffs)

    # Convert complex coefficients to real Fourier coefficients.
    a = {0: sp.simplify(2 * coeffs.get(0, 0))}
    b = {}

    for k in range(1, N + 1):
        ck = coeffs.get(k, 0)
        c_minus_k = coeffs.get(-k, 0)

        a[k] = sp.simplify(ck + c_minus_k)
        b[k] = sp.simplify(I * (ck - c_minus_k))

    return a, b

def fourier_expression(a, b, x):
    """
    Reconstruct a Fourier expression from coefficient dictionaries.

    Fourier convention:
        f(x) = a[0]/2 + sum(
            a[k]*cos(k*x) + b[k]*sin(k*x)
        for k >= 1)

    Parameters
    ----------
    a : dict
        Cosine coefficients, including a[0].
    b : dict
        Sine coefficients.
    x : sympy Symbol
        Independent variable.

    Returns
    -------
    sympy expression
        Reconstructed Fourier expression.
    """
    expr = a.get(0, 0) / 2

    harmonics = set(a.keys()) | set(b.keys())
    harmonics.discard(0)

    for k in sorted(harmonics):
        expr += a.get(k, 0) * sp.cos(k*x)
        expr += b.get(k, 0) * sp.sin(k*x)

    return sp.simplify(expr)

def fourier_rewrite(expr: sp.Expr, x: sp.Symbol) -> sp.Expr:
    return fourier_expression(
        *fourier_coefficients(expr, x), x
    )


def truncated_product_2d(expr1: sp.Expr, 
                         expr2: sp.Expr, 
                         x: sp.Symbol, 
                         y: sp.Symbol, 
                         a: int, 
                         b: int):
    """
    Compute product of two polynomials up to order x^a * y^b.
    Avoids computing unnecessary higher order terms
    """
    P = sp.Poly(sp.expand(expr1), x, y)
    Q = sp.Poly(sp.expand(expr2), x, y)
    
    result = 0

    for i in range(a + 1):
        for j in range(b + 1):
            for k in range(i + 1):
                for l in range(j + 1):
                    result += (
                        P.coeff_monomial(x**k * y**l)
                        * Q.coeff_monomial(x**(i-k) * y**(j-l))
                        * x**i * y**j
                    )

    return sp.expand(result)

if __name__=='__main__':
    from algebraic.mercier_criterion.rendering import function_to_latex

    x, y = sp.symbols('x y')
    n = sp.symbols('n', integer=True, nonnegative=True)
    a0,a1,a2 = sp.symbols('a0 a1 a2')

    expr_1 = a0+a1*x+a2*y
    expr_2 = a0+a1*x*y + a2*x**2

    expanded = truncated_product_2d(
        expr_1, expr_2, x, y, 1, 1
    )
    series_expand = sp.expand(expr_1*expr_2)
    function_to_latex([expanded, series_expand])
    # expanded = sp.expand(expr.subs(sp.sin(x)**2, 1 - sp.cos(x)**2))
    # #expr_sum = trig_power_to_sum(expanded, x)
    # function_to_latex(expanded)

    
