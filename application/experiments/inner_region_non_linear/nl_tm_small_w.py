import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import imports

from tearing_mode_solver.nl_td_solver import solve_time_dependent_system
from tearing_mode_solver.outer_region_solver import (
    delta_prime,
    magnetic_shear
)
from tearing_mode_solver.algebraic_fitting import (
    nl_parabola_coefficients, parabola
)
from tearing_mode_solver.helpers import savefig

def nl_tm_small_w():
    """
    Numerically solve the tearing mode problem in the strongly non-linear regime
    starting with a small seeding island value.

    We then plot this solution as a function of time together with the algebraic
    solution found in the small island limit.
    """
    m=3
    n=2
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e7, 10000)

    td0, tm0 = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    psi_t = td0.psi_t

    psi0 = psi_t[0]
    dp = delta_prime(tm0)
    s = magnetic_shear(tm0.r_s, m, n)

    a_theory, b_theory, c_theory = nl_parabola_coefficients(
        tm0,
        s,
        lundquist_number,
        dp,
        psi0
    )

    print(
        f"""Theoretical fit at^2 + bt + c:
            a={a_theory}, b={b_theory}, c={c_theory}"""
    )

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(times, psi_t, label='Numerical solution', color='black')

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")

    # Returns highest power coefficients first
    fit, cov = curve_fit(
        parabola,
        times,
        psi_t,
        p0=(a_theory, b_theory, c_theory),
        bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        x_scale = (1.0/psi0, 1.0/psi0, 1.0/psi0),
        method='trf'
    )
    perr = np.sqrt(np.diag(cov))
    print("Coefs: ", fit)
    print("Errors: ", perr)
    poly = np.poly1d(fit)


    t_fit = (a_theory, b_theory, c_theory)
    ax.plot(
        times, parabola(times, *t_fit), label='Theoretical poly', linestyle='dotted',
        color='red'
    )
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend(prop={'size': 7})

    ls_data = np.sum((parabola(times, *fit) - psi_t)**2)
    ls_theory = np.sum((parabola(times, *t_fit) - psi_t)**2)

    print(ls_data, ls_theory)

    fig.tight_layout()
    #plt.show()
    savefig(f"nl_small_w_(m,n,A)=({m},{n},{solution_scale_factor})")
    plt.show()

if __name__=='__main__':
    nl_tm_small_w()
