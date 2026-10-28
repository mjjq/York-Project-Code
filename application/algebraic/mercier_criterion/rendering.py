import sympy as sp

from algebraic.mercier_criterion.datatypes import (
    ToroidalCoordinates
)

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

def function_to_latex(f_list: list[sp.Function], 
                      filename: str = 'formula.tex'):
    print(type(f_list))
    if isinstance(f_list, list):
        latex = [sp.latex(f, long_frac_ratio=2) for f in f_list]
    else:
        latex = [sp.latex(f_list, long_frac_ratio=2)]

    #latex = [l.replace("+", r"\allowbreak{}+") for l in latex]

    print(latex)

    # print()
    # print("Jacobian")
    # sp.pprint(sp.simplify(sp.simplify(sp.series(j, coords.epsilon, 0, 2))), use_unicode=True)

    to_latex = r"\end{dmath*}\begin{dmath*}".join(latex)
    document = rf"""
    \documentclass[a4paper]{{article}}
    \usepackage[left=1.5cm, right=1.5cm]{{geometry}}
    \usepackage{{amsmath}}
    \usepackage{{breqn}}
    \pagestyle{{empty}}
    \begin{{document}}
    \footnotesize
    \begin{{dmath*}}
    """+to_latex+rf"""
    \end{{dmath*}}
    \end{{document}}
    """

    with open(filename, "w") as file:
        file.write(document)

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
