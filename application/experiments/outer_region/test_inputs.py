from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
import numpy as np

from tearing_mode_solver.profiles import generate_j_profile, generate_q_profile, dj_dr


def compare_dj_dr_profiles():
    j_prof = generate_j_profile(1.0, 2.0)
    q_prof = generate_q_profile(1.0, 2.0)

    r_vals, j_vals = zip(*j_prof)

    j_func = UnivariateSpline(r_vals, j_vals, s=0.0)
    dj_dr_func = j_func.derivative()

    fig, axs = plt.subplots(2)
    ax, ax2 = axs

    new_r_vals = np.linspace(0.6, 0.7, 100000)

    dj_dr_vals_analytic = dj_dr(new_r_vals, 2.0, 1.0)
    ax.plot(new_r_vals, dj_dr_vals_analytic, color='black', label='analytic')

    dj_dr_vals_numeric = dj_dr_func(new_r_vals)
    ax.plot(new_r_vals, dj_dr_vals_numeric, color='red', label='numeric')

    sq_err = (dj_dr_vals_analytic - dj_dr_vals_numeric)**2

    ax2.plot(new_r_vals, sq_err)

if __name__=='__main__':
    compare_dj_dr_profiles()

    plt.show()