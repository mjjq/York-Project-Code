import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

import imports
from tearing_mode_solver.outer_region_solver import (
    rational_surface, magnetic_shear
)
from tearing_mode_solver.delta_model_solver import nu, mode_width
from tearing_mode_solver.helpers import (
    classFromArgs, TimeDependentSolution, savefig, load_sim_from_disk
)
from tearing_mode_solver.outer_region_solver import island_width

def ql_tm_vs_time():
    """
    Plot various numerically solved variables from a tearing mode solution and
    island width as a function of time from .csv data.
    """
    model_data_filename = "./output/23-04-2024_16:44_jorek_model_(m,n)=(2,1).zip"
    params, sol = load_sim_from_disk(model_data_filename)

    times = sol.times
    psi_t = sol.psi_t
    dpsi_t = sol.dpsi_dt
    w_t = sol.w_t
    d2psi_dt2 = sol.d2psi_dt2
    delta_primes = sol.delta_primes

    #plt.plot(w_t)
    #plt.show()

    #print("lgr: ", lin_growth_rate)
    #print("Threshold: ", ql_threshold)
    #print(psi_t)

    fig, ax = plt.subplots(1, figsize=(4,3))
    ax2 = ax.twinx()

    m = params.poloidal_mode_number
    n = params.toroidal_mode_number
    S = params.lundquist_number
    r_s = rational_surface(
        params.q_profile, 
        m/n
    )
    s=magnetic_shear(params.q_profile, r_s)
    modal_widths = mode_width(
        psi_t,
        dpsi_t,
        d2psi_dt2,
        r_s,
        m,
        n,
        s,
        S
    )

    ax.plot(times, psi_t, label='Flux', color='black')

    ax.set_xlabel(r"Normalised time ($1/\bar{\omega}_A$)")
    ax.set_ylabel(r"Normalised perturbed flux ($a^2 B_{\phi 0}$)")

    ax2.plot(times, modal_widths, label='Normalised island width', color='red')
    ax2.set_ylabel(r"Normalised modal width ($a$)")
    ax2.yaxis.label.set_color('red')

    #ax.legend(prop={'size': 7}, loc='lower right')

    # ax.set_yscale('log')
    # ax2.set_yscale('log')
    ax.set_xscale('log')
    ax2.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_yscale('log')

    fig.tight_layout()
    #plt.show()
    orig_fname, ext = os.path.splitext(os.path.basename(model_data_filename))
    savefig(f"{orig_fname}_full")

    fig_growth, ax_growth = plt.subplots(1, figsize=(4.5,3))

    ax_growth.plot(times, dpsi_t/psi_t, color='black')
    ax_growth.set_ylabel(r'Growth rate $\delta\dot{\psi}^{(1)}/\delta\psi^{(1)}$')
    ax_growth.set_xlabel(r'Normalised time $1/\bar{\omega}_A$')

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig_growth.tight_layout()

    ax_growth.set_xscale('log')
    orig_fname, ext = os.path.splitext(os.path.basename(model_data_filename))
    savefig(f"{orig_fname}_growth_rate")

    ax_growth.set_xscale('log')
    #ax_growth.set_xlim(left=0.0, right=1e5)
    #ax_growth.set_ylim(bottom=5.46e-5, top=5.50e-5)

    fig_growth.tight_layout()

    savefig(f"{orig_fname}_growth_rate_zoomed")



    fig_d2psi, ax_second = plt.subplots(4, sharex=True)
    ax_d2psi,  ax_nl, ax_l, ax_dwdt = ax_second

    fig_d2psi_actual, ax_d2psi = plt.subplots(1, figsize=(4,3))
    ax_d2psi.set_xlabel(r'Normalised time $(\bar{\omega}_A t)$')
    ax_d2psi.plot(times, d2psi_dt2, color='black')
    ax_d2psi.set_ylabel(r'$\ddot{\delta\psi}$')
    ax_second[-1].set_xlabel(r'Normalised time $(\bar{\omega_A} t)$')
    ax_d2psi.set_xscale('log')
    ax_d2psi.set_yscale('log')

    ax_d2psi.set_xscale('linear')
    ax_d2psi.set_yscale('linear')
    #ax_d2psi.set_xlim(left=1e5, right=3e5)
    ax_d2psi.set_ylim(bottom=0.0, top=2e-15)
    fig_d2psi_actual.tight_layout()
    savefig(f"{orig_fname}_d2psi_dt2")




    nl_growth = nu(psi_t, m, S, r_s)
    ax_nl.plot(times, nl_growth, color='black')
    ax_nl.set_xscale('log')
    ax_nl.set_yscale('log')
    ax_nl.set_ylabel(r'$\frac{1}{2} m^2 S (\delta \psi)^2/\hat{r}_s^4}$')


    linear_term = S*(n*s)**2 * (
        delta_primes * r_s * psi_t/(2.12*S*dpsi_t)
    )**4
    ax_l.plot(times, linear_term, color='black')
    ax_l.set_xscale('log')
    ax_l.set_yscale('log')
    ax_l.set_ylabel(r'Linear contribution to $\ddot{\delta\psi}$')

    # TODO: Fix issue with UnivariateSpline not working on some data
    w_t_func = UnivariateSpline(times, w_t, s=0)
    dw_dt_func = w_t_func.derivative()
    w_growth = dw_dt_func(times)/w_t_func(times)

    ax_dwdt.plot(times, w_growth, color='black')
    ax_dwdt.set_yscale('log')
    ax_dwdt.set_ylabel(r'$\dot{\delta}_{ql}/\delta_{ql}$')

    fig_d2psi.tight_layout()

    fig_psis, ax_psis = plt.subplots(5)
    (ax_psi_only, ax_dpsi_dt_only,
     ax_d2psi_only, ax_delql, ax_dprime) = ax_psis

    ax_psi_only.plot(times, psi_t, color='black')
    ax_psi_only.set_yscale('log')
    ax_psi_only.set_ylabel(r'$\delta\psi$')

    ax_dpsi_dt_only.plot(times, dpsi_t, color='black')
    ax_dpsi_dt_only.set_yscale('log')
    ax_dpsi_dt_only.set_ylabel(r'$\dot{\delta\psi}$')

    ax_d2psi_only.plot(times, d2psi_dt2, color='black')
    ax_d2psi_only.set_yscale('log')
    ax_d2psi_only.set_ylabel(r'$\ddot{\delta\psi}$')

    ax_delql.plot(times, w_t, color='black')
    ax_delql.set_ylabel(r"$\delta_{ql}$")
    ax_delql.set_yscale('log')

    ax_dprime.plot(times, delta_primes, color='black')
    ax_dprime.set_ylabel(r"$\Delta'[\delta_{ql}]$")

    ax_psis[-1].set_xlabel(r'Normalised time $(\bar{\omega_A} t)$')




    for ax_p in ax_psis:
        ax_p.set_xscale('log')

    fig_psis.tight_layout()

    plt.show()

if __name__=='__main__':
    ql_tm_vs_time()
