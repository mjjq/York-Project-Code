import numpy as np
from matplotlib import pyplot as plt

from tearing_mode_solver.loizu_delta_prime import calculate_coefficients, delta_prime_loizu
from tearing_mode_solver.profiles import generate_j_profile, generate_q_profile
from tearing_mode_solver.outer_region_solver import TearingModeParameters, delta_prime_non_linear, solve_system


def compare_delta_prime_schemes():
    params = TearingModeParameters(
        2, 1, 1e8, 1e-12, 1.0, 10.0,
        generate_q_profile(1.0, 2.0),
        generate_j_profile(1.0, 2.0)
    )
    sol = solve_system(params)

    coefs = calculate_coefficients(params)

    w_vals = np.logspace(-9, -0.5, 10000)

    delta_primes_l = delta_prime_loizu(w_vals, coefs)
    delta_primes_w = delta_prime_non_linear(sol, w_vals)

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(w_vals, coefs.r_s*delta_primes_w, label="Conventional")
    ax.plot(w_vals, coefs.r_s*delta_primes_l, label="Loizu 2023")

    ax.grid()

    ax.set_xlabel("Island width w/a")
    ax.set_ylabel(r"$r_s \Delta'$")
    ax.legend()
    fig.tight_layout()

def check_convergence():
    from tearing_mode_solver.loizu_delta_prime import w0_coefficient, sigma_prime

    params = TearingModeParameters(
        2, 1, 1e8, 1e-12, 1.0, 10.0,
        generate_q_profile(1.0, 2.0),
        generate_j_profile(1.0, 2.0)
    )
    coefs = calculate_coefficients(params)

    epsilons = np.logspace(-9, -3, 100)
    sigmas = []
    w0s = []
    dpsi_dr = []
    for epsilon in epsilons:
        outer_sol = solve_system(params, r_s_thickness=epsilon)


        sig = sigma_prime(outer_sol, coefs.a)
        w0 = w0_coefficient(sig, coefs.a)

        sigmas.append(sig)
        w0s.append(w0)

        dpsi_dr_avg = (outer_sol.dpsi_dr_backwards[-1] + outer_sol.dpsi_dr_forwards[-1])/outer_sol.psi_forwards[-1]
        dpsi_dr.append(dpsi_dr_avg)

    fig, axs = plt.subplots(3, sharex=True)
    ax_sig, ax_w0, ax_dpsi_dr = axs
    for ax in axs:
        ax.grid()
        ax.set_xscale('log')

    axs[-1].set_xlabel("$\epsilon$")

    ax_sig.plot(epsilons, sigmas)
    ax_sig.set_ylabel("$\Sigma'$")
    ax_w0.plot(epsilons, w0s)
    ax_w0.set_ylabel("$w_0$")
    ax_dpsi_dr.plot(epsilons, dpsi_dr)
    ax_dpsi_dr.set_ylabel("Avg. derivative in psi")


if __name__=='__main__':
    compare_delta_prime_schemes()
    check_convergence()

    plt.show()