import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import imports

from tearing_mode_solver.nl_td_solver import solve_time_dependent_system
from tearing_mode_solver.outer_region_solver import (
    delta_prime, magnetic_shear
)
from tearing_mode_solver.algebraic_fitting import (
    nl_parabola_coefficients, parabola
)
from tearing_mode_solver.helpers import savefig


def algebraic_departure():
    """
    Calculate the difference between the numerical solution to the non-linear
    equations and the algebraic solution found in the small island limit.
    """
    m=3
    n=2
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 5e7, 10000)

    td_sol, tm0 = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    psi_t = td_sol.psi_t

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

    fig, ax = plt.subplots(1, figsize=(4,3.5))
    #ax2 = ax.twinx()

    #ax.plot(times, psi_t, label='Numerical solution', color='black')

    #ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_xlabel(r"Numerical flux solution ($\delta \hat{\psi}^{(1)}$)")
    ax.set_ylabel(r"Fractional error in algebraic solution")

    #ax2.plot(times, w_t, label='Normalised island width', color='red')
    #ax2.set_ylabel(r"Normalised island width ($\hat{w}$)")
    #ax2.yaxis.label.set_color('red')

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

    #ax.plot(
        #times, parabola(times, *fit), label='Order 2 poly fit', linestyle='dashed',
        #color='darkturquoise'
    #)

    t_fit = (a_theory, b_theory, c_theory)
    theoretical_psi = parabola(times, *t_fit)
    fractional_change = np.sqrt(((psi_t-theoretical_psi)/theoretical_psi)**2)

    print(fractional_change.shape)
    print(times.shape)
    print(max(times))
    #ax.plot(psi_t, fractional_change)
    ax.plot(psi_t, fractional_change)
    print(min(psi_t), max(psi_t))

    #ax2 = ax.twiny()
    #ax2.plot(psi_t, psi_t)
    #ax2.cla()

    #ax.plot(
        #times, parabola(times, *t_fit), label='Theoretical poly', linestyle='dotted',
        #color='red'
    #)
    ax.set_xscale('log')
    ax.set_yscale('log')

    #ax.legend(prop={'size': 7})

    ls_data = np.sum((parabola(times, *fit) - psi_t)**2)
    ls_theory = np.sum((parabola(times, *t_fit) - psi_t)**2)

    print(ls_data, ls_theory)

    fig.tight_layout()
    #plt.show()
    savefig(
        f"error_algebraic_sol_(m,n,A)=({m},{n},{solution_scale_factor})"
    )
    plt.show()

if __name__=='__main__':
    algebraic_departure()
