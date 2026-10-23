import sympy as sp

def cos_power_to_sum(expr: sp.Expr, x: sp.Symbol):
    expr = sp.expand(expr)
    replacements = {}

    for power in expr.atoms(sp.Pow):
        base, exponent = power.as_base_exp()

        if exponent.is_Integer and exponent >= 2:
            n = int(exponent)

            if base == sp.cos(x):
                replacements[power] = sum(
                    sp.binomial(n, k) * sp.cos((n - 2*k)*x)
                    for k in range(n + 1)
                ) / 2**n

            elif base == sp.sin(x):
                replacements[power] = sum(
                    (-1)**k * sp.binomial(n, k)
                    * sp.cos((n - 2*k)*x + n*sp.pi/2)
                    for k in range(n + 1)
                ) / 2**n

                replacements[power] = sp.expand_trig(
                    replacements[power]
                )

    return expr.xreplace(replacements)



