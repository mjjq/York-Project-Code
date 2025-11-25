import numpy as np
from matplotlib import pyplot as plt
from copy import copy

from tearing_mode_solver.militello_delta_prime import calculate_coefficients, growth_rate_militello, gamma_solvable
from tearing_mode_solver.profiles import generate_j_profile, generate_q_profile
from tearing_mode_solver.outer_region_solver import TearingModeParameters, delta_prime_non_linear, solve_system, growth_rate_full


def check_convergence():
    params = TearingModeParameters(
        2, 1, 1e8, 1e-12, 1.0, 10.0,
        generate_q_profile(1.0, 3.0),
        generate_j_profile(1.0, 3.0)
    )
    sol = solve_system(params)

    coefs = calculate_coefficients(params)

    print(coefs)

    lundquist_numbers = np.logspace(3, 9, 20)
    grs = []
    furth_grs = [
        growth_rate_full(
            coefs.poloidal_mode_number,
            coefs.toroidal_mode_number,
            lq,
            coefs.r_s,
            coefs.shear_rs,
            coefs.delta_prime
        ) for lq in lundquist_numbers
    ]

    for lq in lundquist_numbers:
        new_coefs = copy(coefs)
        new_coefs.lundquist_number = lq

        grs.append(growth_rate_militello(new_coefs))

    print(grs)

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(lundquist_numbers, furth_grs, label='Furth 1973')
    ax.scatter(lundquist_numbers, grs, marker='x', label='Militello 2004')

    ax.grid()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Lundquist number")
    ax.set_ylabel(r"$\gamma/\omega_A$")
    ax.legend()
    fig.tight_layout()



def test_solve_function():
    params = TearingModeParameters(
        2, 1, 1e8, 1e-12, 1.0, 10.0,
        generate_q_profile(1.0, 2.0),
        generate_j_profile(1.0, 2.0)
    )
    sol = solve_system(params)

    coefs = calculate_coefficients(params)

    print(coefs)

    gamma_guesses = np.logspace(-7, -3, 100)

    fig, ax = plt.subplots(1, figsize=(4,3))

    lundquist_numbers = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]

    for lq in lundquist_numbers:
        coefs.lundquist_number = lq
        ret = []

        furth_gamma = growth_rate_full(
            coefs.poloidal_mode_number,
            coefs.toroidal_mode_number,
            coefs.lundquist_number,
            coefs.r_s,
            coefs.shear_rs,
            coefs.delta_prime
        )
        print(furth_gamma)

        for guess in gamma_guesses:
            s = gamma_solvable(guess, coefs)

            ret.append(s)

        ax.plot(gamma_guesses, ret, label=f"S={lq:.2g}")
        ax.scatter(furth_gamma, 0.0)

    ax.grid()

    ax.set_xscale('log')
    ax.set_ylim(top=10.0, bottom=-10.0)
    ax.set_xlabel("Growth rate")
    ax.set_ylabel(r"LHS-RHS")
    ax.legend()
    fig.tight_layout()



if __name__=='__main__':
    #test_solve_function()
    check_convergence()

    plt.show()