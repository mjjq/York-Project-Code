import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

def ql_tm_vs_time():
    fname = "./output/17-08-2023_17:06_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)

    times = df['times']
    psi_t = df['psi_t']
    dpsi_t = df['dpsi_dt']
    w_t = df['w_t']
    d2psi_dt2 = df['d2psi_dt2']

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
    ax_d2psi,  ax_nl, ax_dwdt, ax_d2psi_only = ax_second
    ax_d2psi.plot(times, d2psi_dt2/dpsi_t, color='black')
    ax_d2psi.set_ylabel(r'$\left( \dot{\delta\psi} \right)^{-1} \ddot{\delta\psi}$')
    ax_second[-1].set_xlabel(r'Normalised time $(\bar{\omega_A} t)$')
    ax_d2psi.set_xscale('log')
    ax_d2psi.set_yscale('log')

    ax_d2psi_only.plot(times, d2psi_dt2, color='black')
    ax_d2psi_only.set_yscale('log')
    ax_d2psi_only.set_ylabel(r'$\ddot{\delta\psi}$')

    m=2
    S=1e8
    r_s = 0.79
    nl_growth = (0.5*m**2)*S*(psi_t**2)/r_s**4
    ax_nl.plot(times, nl_growth, color='black')
    ax_nl.set_xscale('log')
    ax_nl.set_yscale('log')
    ax_nl.set_ylabel(r'$\frac{1}{2} m^2 S (\delta \psi)^2/\hat{r}_s^4}$')

    w_t_func = UnivariateSpline(times, w_t)
    dw_dt_func = w_t_func.derivative()
    w_growth = dw_dt_func(times)/w_t_func(times)

    ax_dwdt.plot(times, w_growth, color='black')
    ax_dwdt.set_yscale('log')
    ax_dwdt.set_ylabel(r'$\dot{\delta}_{ql}/\delta_{ql}$')

    fig_d2psi.tight_layout()
    plt.show()


if __name__=='__main__':
    ql_tm_vs_time()
