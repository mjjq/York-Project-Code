import numpy as np
from matplotlib import pyplot as plt

import imports

from tearing_mode_solver.gamma_model_solver import (
    solve_time_dependent_system,
    time_from_flux
)
from tearing_mode_solver.outer_region_solver import growth_rate
from tearing_mode_solver.algebraic_fitting import nl_parabola
from tearing_mode_solver.helpers import savefig

def ql_with_fit_plots():
    """
    Deprecated function.
    """
    m=4
    n=3
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e2, 1000)

    ql_solution, tm, ql_threshold, s = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )
    times = ql_solution.times
    psi_t = ql_solution.psi_t
    dpsi_t = ql_solution.dpsi_dt
    w_t = ql_solution.w_t
    dps = ql_solution.delta_primes

    lin_delta_prime, lin_growth_rate = growth_rate(
        m,
        n,
        lundquist_number,
        axis_q
    )

    #print("lgr: ", lin_growth_rate)
    #print("Threshold: ", ql_threshold)
    #print(psi_t)

    fig, ax = plt.subplots(1, figsize=(4,3))

    ql_threshold = 1.0 # TODO: Re-add correct threshold
    ql_time_min = time_from_flux(psi_t, times, 0.1*ql_threshold)
    ql_time_max = time_from_flux(psi_t, times, 10.0*ql_threshold)

    ax.fill_between(
        [ql_time_min, ql_time_max],
        2*[min(psi_t)],
        2*[max(psi_t)],
        alpha=0.3,
        label='Quasi-linear region'
    )

    ax.plot(times, psi_t, label='Flux', color='black')

    lin_times = times[np.where(times < ql_time_max)]
    ax.plot(
        lin_times,
        psi_t[0]*np.exp(lin_growth_rate*lin_times),
        label='Exponential fit'
    )

    nl_times = times[np.where(times >= ql_time_max)]
    psi_t0_nl = psi_t[np.abs(times-ql_time_max).argmin()]
    dp_nl = dps[np.abs(times-ql_time_max).argmin()]
    print(psi_t0_nl)
    #ax.plot(
        #nl_times,
        #nl_parabola(
            #tm0,
            #s,
            #lundquist_number,
            #dp_nl,
            #psi_t0_nl,
            #nl_times
        #),
        #label="Quadratic fit"
    #)



    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")

    ax.legend(prop={'size': 7}, loc='lower right')

    # ax.set_yscale('log')
    # ax2.set_yscale('log')
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()
    #plt.show()
    savefig(
        f"ql_with_fitting_(m,n,A)=({m},{n},{solution_scale_factor})"
    )
    plt.show()

if __name__=='__main__':
    ql_with_fit_plots()
