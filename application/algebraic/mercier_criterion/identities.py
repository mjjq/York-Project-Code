import sympy as sp
import re
from sympy.core.function import AppliedUndef

def collect_cos_coefs(expr: sp.Expr):
    """
    Collect all coefficients of cosine harmonics
    e.g. A + B*cos(x) + C*cos(2x) + ...
    """
    expanded = sp.expand(expr)
    cos_terms = sorted(
        expanded.atoms(sp.cos),
        key=sp.default_sort_key
    )
    return sp.collect(expanded, cos_terms)

def get_cos_coefs(expr: sp.Expr, 
                  x: sp.Symbol, 
                  indices: list[int]) -> list[sp.Expr]:
    basis = []
    for index in indices:
        if index!=0:
            basis.append(sp.cos(index*x))

    coefs = [expr.coeff(term) for term in basis]

    if 0 in indices:
        constant, variable_part = expr.as_independent(x, as_Add=True)
        coefs.insert(0, constant)
    
    return coefs

def sin2n_to_cos2n(expr: sp.Expr, x: sp.Symbol):
    """
    Replace sin^(2n)(x) with (1-cos^2(x))^n
    """
    s = sp.sin(x)
    c = sp.cos(x)

    expr = expr.replace(
        lambda e: (
            e.is_Pow
            and e.base == s
            and e.exp.is_even
            and e.exp.is_Integer
            and e.exp > 0
        ),
        lambda e: (1 - c**2)**(e.exp // 2)
    )

    return sp.expand(expr)

def trig_power_to_sum(expr: sp.Expr, x: sp.Symbol):
    """
    Expands sin^n(x) or cos^n(x) in terms of sums of
    cos(nx) (or sin(nx) if n is odd for sin^n(x)).

    Any powers of sin^
    """
    expr = sp.expand(expr)

    replacements = {}

    for power in expr.atoms(sp.Pow):
        base, exponent = power.as_base_exp()

        if exponent.is_Integer and exponent >= 2:
            n = int(exponent)

            if base == sp.cos(x):
                replacements[power] = sp.expand(
                    sum(
                        sp.binomial(n, k) * sp.cos((n - 2*k)*x)
                        for k in range(n + 1)
                    ) / 2**n
                )

            elif base == sp.sin(x):
                replacements[power] = sp.expand(
                    sum(
                        (-1)**k * sp.binomial(n, k)
                        * sp.cos((n - 2*k)*x - n*sp.pi/2)
                        for k in range(n + 1)
                    ) / 2**n
                )

    return expr.xreplace(replacements)

def expand_sin_terms(expr: sp.Expr, x: sp.Symbol) -> sp.Expr:
    """
    Expand sin(mx) terms in expression to sum of sin^n(x), cos^n(x)
    terms while leaving cos(mx) terms untouched.
    """
    replacements = {}
    
    for sin_term in expr.atoms(sp.sin):
        arg = sin_term.args[0]
        m = arg.coeff(x)

        if m.is_Integer and m >= 2:
            replacements[sin_term] = sum(
                (-1)**k * sp.binomial(m, 2*k+1)
                * sp.sin(x)**(2*k + 1)
                * sp.cos(x)**(m-2*k-1)
                for k in range((m+1)//2)
            )

    return expr.xreplace(replacements)


def expand_cos_terms(expr: sp.Expr, x: sp.Symbol) -> sp.Expr:
    """
    Expand cos(mx) terms in expression to sum of sin^n(x), cos^n(x)
    terms while leaving sin(mx) terms untouched.
    """
    replacements = {}

    for cos_term in expr.atoms(sp.cos):
        arg = cos_term.args[0]
        m = arg.coeff(x)

        if m.is_Integer and m >= 2 and arg == m*x:
            replacements[cos_term] = sum(
                (-1)**k * sp.binomial(m, 2*k)
                * sp.sin(x)**(2*k)
                * sp.cos(x)**(m - 2*k)
                for k in range(m // 2 + 1)
            )

    return expr.xreplace(replacements)


def remove_common_factors(expr: sp.Expr) -> sp.Expr:
    common_factor = sp.gcd_list(sp.Add.make_args(expr))

    return sp.simplify(expr/common_factor)


def total_derivative_order(term: sp.Expr) -> int:
    order = 0
    for factor in sp.Mul.make_args(term):
        base, exp = factor.as_base_exp()
        if isinstance(base, sp.Derivative):
            order += base.derivative_count * int(exp)
    return order

def remove_derivative_order(expr: sp.Expr, max_order: int = 2) -> sp.Expr:
    """
    Remove terms in an expression whose derivative
    order is larger than max_order
    """
    return sp.Add(*[
        term for term in sp.expand(expr).as_ordered_terms()
        if total_derivative_order(term) <= max_order
    ])


def _scaled_function(name: str, n: int, r: sp.Symbol) -> sp.Expr:
    """Placeholder for the n-th 'prime' derivative F^(n)(r) = eps^-n d^nF/dr^n"""
    return sp.Function(f"{name}_{n}")(r)


def scale_r_derivatives(expr: sp.Expr,
                        r: sp.Symbol,
                        epsilon: sp.Symbol) -> sp.Expr:
    """
    Replace d^nF/dr^n with epsilon**n * F_n(r), where F_n is the
    n-th derivative in Lutjens' prime notation (' = (1/eps) d/dr).

    After this, every derivative carries its epsilon cost explicitly,
    so truncating in epsilon also truncates by derivative order.

    Call this AFTER all sp.diff(..., r) calls are finished: F_n(r)
    is a new undefined function, so differentiating it later would
    not know that dF_n/dr = eps * F_{n+1}.
    """
    replacements = {}

    for d in expr.atoms(sp.Derivative):
        # Only pure r-derivatives of an undefined function F(r)
        if any(var != r for var in d.variables):
            continue
        if not isinstance(d.expr, AppliedUndef) or d.expr.args != (r,):
            continue

        n = d.derivative_count
        name = d.expr.func.__name__
        replacements[d] = epsilon**n * _scaled_function(name, n, r)

    return expr.xreplace(replacements)


def restore_r_derivatives(expr: sp.Expr,
                          r: sp.Symbol,
                          epsilon: sp.Symbol,
                          as_primes: bool = False) -> sp.Expr:
    """
    Inverse of scale_r_derivatives.

    as_primes=False: F_n(r) -> eps**(-n) * d^nF/dr^n  (exact inverse)
    as_primes=True:  F_n(r) -> d^nF/dr^n, i.e. treat the Derivative
                     objects as Lutjens' primes (use this for rendering
                     the final equations once epsilon has been divided out)
    """
    pattern = re.compile(r"^(.+)_(\d+)$")
    replacements = {}

    for f in expr.atoms(AppliedUndef):
        m = pattern.match(f.func.__name__)
        if not m or f.args != (r,):
            continue
        name, n = m.group(1), int(m.group(2))
        deriv = sp.Derivative(sp.Function(name)(r), (r, n))
        replacements[f] = deriv if as_primes else deriv / epsilon**n

    return expr.xreplace(replacements)

if __name__=='__main__':
    from algebraic.mercier_criterion.rendering import function_to_latex

    # x = sp.symbols('x')
    # n = sp.symbols('n', integer=True, nonnegative=True)

    # expr = sp.sin(x)**(4) + sp.sin(x)**6
    # expanded = sp.expand(expr.subs(sp.sin(x)**2, 1 - sp.cos(x)**2))
    # #expr_sum = trig_power_to_sum(expanded, x)
    # function_to_latex(expanded)

    a0,a1,a2,a3 = sp.symbols('a0 a1 a2 a3')
    x = sp.Symbol('x')
    f = sp.Function('f')

    expr = (
        x + sp.diff(f(x), x) + sp.diff(sp.diff(f(x), x), x)
        + sp.diff(f(x), x) * sp.diff(sp.diff(f(x), x), x)
    )

    filtered = remove_derivative_order(expr, 1)

    function_to_latex(filtered)
