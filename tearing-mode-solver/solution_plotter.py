import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

from linear_solver import rational_surface, magnetic_shear
from new_ql_solver import nu, mode_width
from helpers import classFromArgs, TimeDependentSolution, savefig
from non_linear_solver import island_width

def ql_tm_vs_time():
    """
    Plot various numerically solved variables from a tearing mode solution and
    island width as a function of time from .csv data.
    """
    fname = "./output/29-08-2023_10:23_new_ql_tm_time_evo_(m,n,A,q0)=(2,1,1e-10,1.0).csv"
    df = pd.read_csv(fname)

    times = df['times']
    psi_t = df['psi_t']
    dpsi_t = df['dpsi_dt']
    w_t = df['w_t']
    d2psi_dt2 = df['d2psi_dt2']
    delta_primes = df['delta_primes']

    #plt.plot(w_t)
    #plt.show()

    #print("lgr: ", lin_growth_rate)
    #print("Threshold: ", ql_threshold)
    #print(psi_t)

    fig, ax = plt.subplots(1, figsize=(4,3))
    ax2 = ax.twinx()


    ax.plot(times, psi_t, label='Flux', color='black')

    ax.set_xlabel(r"Normalised time ($1/\bar{\omega}_A$)")
    ax.set_ylabel(r"Normalised perturbed flux ($a^2 B_{\phi 0}$)")

    ax2.plot(times, w_t, label='Normalised island width', color='red')
    ax2.set_ylabel(r"Normalised modal width ($a$)")
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

    fig_growth, ax_growth = plt.subplots(1, figsize=(4.5,3))

    ax_growth.plot(times, dpsi_t/psi_t, color='black')
    ax_growth.set_ylabel(r'Growth rate $\delta\dot{\psi}^{(1)}/\delta\psi^{(1)}$')
    ax_growth.set_xlabel(r'Normalised time $1/\bar{\omega}_A$')

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig_growth.tight_layout()

    ax_growth.set_xscale('log')
    orig_fname, ext = os.path.splitext(os.path.basename(fname))
    savefig(f"{orig_fname}_growth_rate")

    ax_growth.set_xscale('linear')
    ax_growth.set_xlim(left=0.0, right=1e5)
    ax_growth.set_ylim(bottom=5.46e-5, top=5.50e-5)

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
    ax_d2psi.set_xlim(left=1e5, right=3e5)
    ax_d2psi.set_ylim(bottom=0.0, top=2e-15)
    fig_d2psi_actual.tight_layout()
    savefig(f"{orig_fname}_d2psi_dt2")


    m=2
    n=1
    S=1e8
    r_s = rational_surface(m, n)
    s=magnetic_shear(r_s, m, n)

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

def compare_ql_evolution():
    """
    Plot numerical solutions for a tearing mode solved using the gamma model
    and delta model and compare them.
    """
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname_new = "./output/29-08-2023_10:23_new_ql_tm_time_evo_(m,n,A,q0)=(2,1,1e-10,1.0).csv"
    df_new = pd.read_csv(fname_new)
    ql_sol_new = classFromArgs(TimeDependentSolution, df_new)

    fname_approx = "./output/29-08-2023_10:24_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df_approx = pd.read_csv(fname_approx)
    ql_sol_approx = classFromArgs(TimeDependentSolution, df_approx)

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(
        ql_sol_approx.times,
        ql_sol_approx.psi_t,
        label="Gamma model",
        color='black'
    )
    ax.plot(
        ql_sol_new.times,
        ql_sol_new.psi_t,
        label="Delta model",
        color='red',
        linestyle='--'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    ax.set_ylabel(r"Perturbed flux")

    ax.legend(prop={'size':7})
    fig.tight_layout()

    savefig("delta_vs_fast_approx_psi")

    ax.set_xscale('linear')
    ax.set_xlim(left=0.0, right=1e5)

    savefig("delta_vs_fast_approx_psi_lin")

    ax.set_xlim(left=8e4, right=2e5)

    savefig("delta_vs_fast_approx_psi_lin_ql_region")


    fig2, ax2 = plt.subplots(1, figsize=(4.5,3.5))


    ax2.plot(
        ql_sol_approx.times,
        ql_sol_approx.dpsi_dt/ql_sol_approx.psi_t,
        label="Gamma model",
        color='black'
    )
    ax2.plot(
        ql_sol_new.times,
        ql_sol_new.dpsi_dt/ql_sol_new.psi_t,
        label="Delta model",
        color='red',
        linestyle='--'
    )

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    ax2.set_ylabel(r"Growth rate $\delta\dot{\psi}^{(1)}/\delta\psi^{(1)}$")

    ax2.legend(prop={'size':7})
    fig2.tight_layout()

    savefig("delta_vs_fast_approx_dpsi_dt")

    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.set_xlim(left=0.0, right=1e5)
    ax2.set_ylim(bottom=1e-5, top=7e-5)

    fig2.tight_layout()

    savefig("delta_vs_fast_approx_dpsi_dt_lin")

    ax2.set_xlim(left=8e4, right=2e5)
    ax2.set_ylim(bottom=1e-5, top=7e-5)

    fig2.tight_layout()

    savefig("delta_vs_fast_approx_dpsi_dt_lin_ql_region")



    fig5, ax5 = plt.subplots(1, figsize=(4.5,3.5))


    ax5.plot(
        ql_sol_approx.times,
        ql_sol_approx.d2psi_dt2/ql_sol_approx.dpsi_dt,
        label="Gamma model",
        color='black'
    )
    ax5.plot(
        ql_sol_new.times,
        ql_sol_new.d2psi_dt2/ql_sol_new.dpsi_dt,
        label="Delta model",
        color='red',
        linestyle='--'
    )

    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    ax5.set_ylabel(r"Growth rate $\delta\ddot{\psi}^{(1)}/\delta\dot{\psi}^{(1)}$")

    ax5.legend(prop={'size':7})
    fig5.tight_layout()

    savefig("delta_vs_fast_approx_d2psi_dt2")

    ax5.set_xscale('linear')
    ax5.set_yscale('linear')
    ax5.set_xlim(left=0.0, right=3e5)
    ax5.set_ylim(bottom=0e-5, top=10e-5)

    fig5.tight_layout()

    savefig("delta_vs_fast_approx_d2psi_dt2_lin")



    #fig3, ax3 = plt.subplots(1, figsize=(4,3))

    #ax3.plot(ql_sol_new.times, ql_sol_new.delta_primes, label="Delta model")
    #ax3.plot(ql_sol_approx.times, ql_sol_approx.delta_primes, label="Gamma model")

    ##ax3.set_xscale('log')
    ##ax3.set_yscale('log')
    #ax3.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    #ax3.set_ylabel(r"$a\Delta'$")
    #ax3.legend()



    #fig4, ax4 = plt.subplots(1, figsize=(4,3))

    #ax4.plot(ql_sol_new.times, ql_sol_new.w_t, label="Delta model")
    #ax4.plot(ql_sol_approx.times, ql_sol_approx.w_t, label="Gamma model")

    #ax4.set_xlim(left=0.0, right=1e5)
    #ax4.set_ylim(bottom=0.0, top=1e-3)
    ##ax3.set_xscale('log')
    ##ax3.set_yscale('log')
    #ax4.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    #ax4.set_ylabel(r"Layer width")
    #ax4.legend()

    plt.show()

def difference_in_flux_models():
    """
    Load numerical solutions to a tearing mode calculated using the gamma
    and delta models, calculate the absolute difference in the perturbed flux
    as a function of time, then plot the difference.
    """
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname_new = "./output/28-08-2023_19:29_new_ql_tm_time_evo_(m,n,A,q0)=(2,1,1e-10,1.0).csv"
    df_new = pd.read_csv(fname_new)
    ql_sol_new = classFromArgs(TimeDependentSolution, df_new)

    fname_approx = "./output/28-08-2023_19:36_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df_approx = pd.read_csv(fname_approx)
    ql_sol_approx = classFromArgs(TimeDependentSolution, df_approx)

    psi_t_new_func = UnivariateSpline(
        ql_sol_new.times, ql_sol_new.psi_t, s=0
    )
    psi_t_approx_func = UnivariateSpline(
        ql_sol_approx.times, ql_sol_approx.psi_t, s=0
    )

    times = ql_sol_new.times
    pc_delta = 100.0*(psi_t_approx_func(times)/psi_t_new_func(times) - 1.0)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(times, pc_delta, color='black')
    ax.set_xscale('log')
    ax.set_xlabel(r'Normalised time $\bar{\omega}_A t$')
    ax.set_ylabel(r'Flux change from gamma to delta model (%)')
    fig.tight_layout()
    savefig("flux_delta")

    fig2, ax2 = plt.subplots(1)
    ax2.plot(times, psi_t_new_func(times))
    ax2.plot(times, psi_t_approx_func(times))
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.show()

def ql_modal_width_and_island_width():
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname_new = "./output/29-08-2023_09:55_new_ql_tm_time_evo_(m,n,A,q0)=(2,1,1e-10,1.0).csv"
    df_new = pd.read_csv(fname_new)
    ql_sol = classFromArgs(TimeDependentSolution, df_new)

    island_widths = island_width(
        ql_sol.psi_t,
        r_s,
        m,
        n,
        s
    )

    modal_widths = mode_width(
        ql_sol.psi_t,
        ql_sol.dpsi_dt,
        ql_sol.d2psi_dt2,
        r_s,
        m,
        n,
        s,
        S
    )
    modal_widths = modal_widths*2**(9/4) * r_s

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(ql_sol.times, island_widths,
        label=r'Magnetic island width $[w(t)]$'
    )
    ax.plot(ql_sol.times, modal_widths,
        label=r'Modal width $[2^{9/4}r_s \bar{\delta}_{ql}(t)]$'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Width (fraction of $a$)")

    fig.tight_layout()

    plt.show()


if __name__=='__main__':
    ql_tm_vs_time()
    #compare_ql_evolution()
    #difference_in_flux_models()
    #ql_modal_width_and_island_width()
