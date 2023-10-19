import numpy as np
from matplotlib import pyplot as plt

import imports
from tearing_mode_solver.gamma_model_solver import (
    solve_time_dependent_system,
    time_from_flux
)
from tearing_mode_solver.outer_region_solver import growth_rate
from tearing_mode_solver.helpers import (
    savefig, sim_to_disk,
    TearingModeParameters
)

def ql_tm_vs_time():
    """
    Numerically solve the quasi-linear time-dependent tearing mode problem
    and plot the perturbed flux and layer width as functions of time.
    """
    params = TearingModeParameters(
        poloidal_mode_number = 2,
        toroidal_mode_number = 1,
        lundquist_number = 1e8,
        axis_q = 1.0,
        profile_shaping_factor = 2.0,
        initial_flux = 1e-10
    )

    times = np.linspace(0.0, 6e7, 100000)

    sol = solve_time_dependent_system(
        params, times
    )

    psi_t, w_t, dps = sol.psi_t, sol.w_t, sol.delta_primes

    lin_delta_prime, lin_growth_rate = growth_rate(
        params.poloidal_mode_number,
        params.toroidal_mode_number,
        params.lundquist_number,
        params.axis_q,
        params.profile_shaping_factor
    )

    #print("lgr: ", lin_growth_rate)
    #print("Threshold: ", ql_threshold)
    #print(psi_t)

    fig, ax = plt.subplots(1, figsize=(4,3))
    ax2 = ax.twinx()

    ql_threshold = 1.0
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

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")

    ax2.plot(times, w_t, label='Normalised island width', color='red')
    ax2.set_ylabel(r"Normalised layer width ($\hat{\delta}$)")
    ax2.yaxis.label.set_color('red')

    ax.legend(prop={'size': 7}, loc='lower right')

    # ax.set_yscale('log')
    # ax2.set_yscale('log')
    ax.set_xscale('log')
    ax2.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_yscale('log')

    fig.tight_layout()
    #plt.show()
    fname = f"gamma_model_(m,n,A)=({params.poloidal_mode_number},{params.toroidal_mode_number},{params.initial_flux})"
    savefig(fname)
    sim_to_disk(fname, params, sol)
    plt.show()

if __name__=='__main__':
    ql_tm_vs_time()
