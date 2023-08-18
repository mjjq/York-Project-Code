import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from linear_solver import rational_surface, magnetic_shear
from new_ql_solver import nu

def ql_tm_vs_time():
    fname = "./output/18-08-2023_16:41_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
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


    fig_d2psi, ax_second = plt.subplots(4, sharex=True)
    ax_d2psi,  ax_nl, ax_l, ax_dwdt = ax_second
    ax_d2psi.plot(times, d2psi_dt2/dpsi_t, color='black')
    ax_d2psi.set_ylabel(r'$\left( \dot{\delta\psi} \right)^{-1} \ddot{\delta\psi}$')
    ax_second[-1].set_xlabel(r'Normalised time $(\bar{\omega_A} t)$')
    ax_d2psi.set_xscale('log')
    ax_d2psi.set_yscale('log')



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


if __name__=='__main__':
    ql_tm_vs_time()
