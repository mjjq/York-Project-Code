import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import imports

from tearing_mode_solver.nl_td_solver import solve_time_dependent_system
from tearing_mode_solver.outer_region_solver import (
    delta_prime, magnetic_shear
)
from tearing_mode_solver.helpers import savefig

def const_psi_approximation():
    """
    Calculate layer_width * Delta' and plot as a function of layer width
    to test the constant-psi approximation for a strongly non-linear solution.

    The constant-psi approximation breaks down if layer_width*Delta' ~ 1
    """
    m=2
    n=1
    lundquist_number = 1e8
    axis_q = 1.2
    solution_scale_factor = 1e-5

    times = np.linspace(0.0, 1e8, 10000)

    td_sol, tm0 = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    d_delta = td_sol.delta_primes * td_sol.w_t

    fig, ax = plt.subplots(1)

    ax.plot(td_sol.w_t, d_delta, color='black')

    ax.set_xlabel(r"Layer width")
    ax.set_ylabel(r"$\delta \Delta'$")

    fig.tight_layout()
    #plt.show()
    savefig(
        f"nl_const_psi_approx_(m,n,A)=({m},{n},{solution_scale_factor})"
    )
    plt.show()

if __name__=='__main__':
    const_psi_approximation()
