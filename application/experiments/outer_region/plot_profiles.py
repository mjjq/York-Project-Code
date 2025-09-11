from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

from tearing_mode_solver.helpers import savefig


def plot_profile_new(r_vals: np.array,
                     shaping_exponent: float):
    from tearing_mode_solver.profiles import (
        generate_j_profile, generate_q_profile,
        rational_surface
    )

    j_profile = generate_j_profile(1.0, shaping_exponent)

    r_vals, j_vals = zip(*j_profile)
    j_func = UnivariateSpline(r_vals, j_vals, s=0.0)
    dj_dr_func = j_func.derivative()
    j_deriv = dj_dr_func(r_vals)

    q_prof = generate_q_profile(1.0, shaping_exponent)
    r_q_vals, q_vals = zip(*q_prof)

    r_s = rational_surface(q_prof, 2.0)

    fig, ax = plt.subplots(2)
    ax_djdr, ax_q = ax

    ax_djdr.plot(r_vals, j_deriv)
    ax_djdr.vlines(r_s, min(j_deriv), max(j_deriv), color='red')
    ax_djdr.grid()

    ax_q.plot(r_q_vals, q_vals)
    ax_q.vlines(r_s, min(q_vals), max(q_vals), color='red')
    ax_q.grid()

    savefig("profiles_new")

    for r, dj_dr_val in zip(r_vals, j_deriv):
        print(r, dj_dr_val)

def plot_profile_old(r_vals: np.array,
                     shaping_exponent: float):
    from tearing_mode_solver.profiles import q, dj_dr, rational_surface

    rs = rational_surface(2.0, shaping_exponent)

    j_deriv = dj_dr(r_vals, shaping_exponent)
    q_prof = q(r_vals, shaping_exponent)

    fig, ax = plt.subplots(2)
    ax_djdr, ax_q = ax

    ax_djdr.plot(r_vals, j_deriv)
    ax_djdr.vlines(rs, min(j_deriv), max(j_deriv), color='red')
    ax_djdr.grid()

    ax_q.plot(r_vals, q_prof)
    ax_q.vlines(rs, min(q_prof), max(q_prof), color='red')
    ax_q.grid()

    savefig("profiles_old")

    for r, dj_dr_val in zip(r_vals, j_deriv):
        print(r, dj_dr_val)


if __name__=='__main__':
    r_vals = np.linspace(0.0, 1.0, 1000)
    shaping_exponent = 2.0

    plot_profile_new(r_vals, shaping_exponent)

    plt.show()