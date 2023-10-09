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
    ax.set_xlabel(r"Normalised time ($1/\bar{\omega}_A$)")
    ax.set_ylabel(r"Perturbed flux ($a^2 B_{\phi 0}$)")

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
    ax2.set_xlabel(r"Normalised time ($1/\bar{\omega}_A$)")
    ax2.set_ylabel(r"Growth rate $\delta\dot{\psi}^{(1)}/\delta\psi^{(1)}$ ($\bar{\omega}_A$)")

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
    ax5.set_xlabel(r"Normalised time $1/\bar{\omega}_A$")
    ax5.set_ylabel(r"Growth rate $\delta\ddot{\psi}^{(1)}/\delta\dot{\psi}^{(1)}$ ($\bar{\omega}_A$)")

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
