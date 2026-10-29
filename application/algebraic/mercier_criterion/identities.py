import sympy as sp
import re
from sympy.core.function import AppliedUndef

def integrate_by_parts(u: sp.Expr, dv: sp.Expr, x: sp.Symbol) -> sp.Expr:
    v = sp.integrate(dv, x)
    du = sp.diff(u, x)

    return u*v - sp.integrate(v*du, x)

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

if __name__=='__main__':
    from algebraic.mercier_criterion.rendering import function_to_latex

    x = sp.symbols('x')
    n = sp.symbols('n', integer=True, nonnegative=True)
    a0,a1,a2 = sp.symbols('a0 a1 a2')

    expr = sp.sin(2*x)**2 + 2*sp.sin(3*x)**2 * sp.cos(3*x)

    coefs = fourier_coefficients(expr, x)
    new_expr = fourier_expression(*coefs, x)
    function_to_latex(new_expr)
    # expanded = sp.expand(expr.subs(sp.sin(x)**2, 1 - sp.cos(x)**2))
    # #expr_sum = trig_power_to_sum(expanded, x)
    # function_to_latex(expanded)

    
