import numpy as np
from matplotlib import pyplot as plt

from lmfit.models import ExponentialModel

import imports
from tearing_mode_solver.gamma_model_solver import (
    solve_time_dependent_system,
    time_from_flux
)
from tearing_mode_solver.outer_region_solver import growth_rate
from tearing_mode_solver.helpers import savefig

def check_exponential_fit():
    """
    Solve quasi-linear tearing mode numerically and calculate RMS error in the
    exponential fit in the linear regime for increasing time intervals.
    """
    m=3
    n=2
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e8, 10000)

    sol, tm0, ql_threshold, s = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    psi_t, w_t, dps = sol.psi_t, sol.w_t, sol.delta_primes

    lin_delta_prime, lin_growth_rate = growth_rate(
        m,
        n,
        lundquist_number,
        axis_q
    )

    # Estimate time at which quasi-linear region begins. This is the time at
    # which the perturbed flux is 1/10th the amount of the threshold flux.
    ql_time_min = time_from_flux(psi_t, times, 0.1*ql_threshold)
    # Estimate time at which quasi-linear region ends and strongly non-linear
    # regime begins. This is 10 times the amount of the threshold flux.
    ql_time_max = time_from_flux(psi_t, times, 10.0*ql_threshold)

    # Define an array of upper values for the fitting interval.
    max_times = np.linspace(0.5*ql_time_min, 2.0*ql_time_max, 100)

    fig, ax = plt.subplots(1, figsize=(4,3))

    chisqrs = []

    for max_time in max_times:
        # Grab only the data in the interval 0 <= t <= max_time
        lin_filter = np.where(times < max_time)
        lin_times = times[lin_filter]
        lin_psi = psi_t[lin_filter]

        #linear_model = ExponentialModel()
        #params = linear_model.make_params(
        #    amplitude=psi_t[0],
        #    decay=-1.0/lin_growth_rate
        #)
        #result = linear_model.fit(lin_psi, params, x=lin_times)

        exp_fit = lin_psi[0]*np.exp(lin_growth_rate*lin_times)

        rms_frac_error = np.mean((1.0-lin_psi/exp_fit)**2)
        chisqrs.append(rms_frac_error)

        #chisqrs.append(result.chisqr)

    print(chisqrs)
    ax.plot(max_times, chisqrs, color='black')

    ax.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    ax.set_ylabel(r"Exponential fit RMS fractional error")

    #ax.set_xlim(left=0.0, right=2.0*ql_time_min)
    #ax.set_ylim(top=1.0)
    ax.set_yscale('log')
    ax.grid(which='major')

    ax.fill_between(
        [ql_time_min, ql_time_max],
        -200.0,
        200.0,
        alpha=0.3,
        label='Quasi-linear region'
    )

    ax.legend()

    ax.set_ylim(top=1.0)

    fig.tight_layout()

    savefig(
        f"frac_error_exp_fit_(m,n,A)=({m},{n},{solution_scale_factor})"
    )

    plt.show()

if __name__=='__main__':
    check_exponential_fit()
