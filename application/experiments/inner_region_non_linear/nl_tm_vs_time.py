def nl_tm_vs_time():
    """
    Calculate tearing mode solution in strongly non-linear regime and plot
    the solution as a function of time.
    """
    m=2
    n=1
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 0.3

    times = np.linspace(0.0, 1e8, 10000)

    td_sol, tm0 = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    psi_t, w_t, delta_primes = td_sol.psi_t, td_sol.w_t, td_sol.delta_primes

    print(psi_t)

    fig, ax = plt.subplots(1, figsize=(4,3))
    ax2 = ax.twinx()

    ax.plot(times, psi_t, label='Normalised perturbed flux', color='black')

    ax.set_xlabel(r"Normalised time ($1/\bar{\omega}_A$)")
    ax.set_ylabel(r"Normalised perturbed flux ($a^2 B_{\phi 0}$)")

    ax2.plot(times, w_t, label='Normalised island width', color='red', linestyle='--')
    ax2.set_ylabel(r"Normalised island width ($a$)")
    ax2.yaxis.label.set_color('red')

    #ax.set_yscale('log')
    #ax2.set_yscale('log')
    #ax.set_xscale('log')
    #ax2.set_xscale('log')

    fig.tight_layout()
    #plt.show()
    fname = f"nl_tm_time_evo_(m,n,A)=({m},{n},{solution_scale_factor})"
    savefig(
        fname
    )
    dataclass_to_disk(fname, td_sol)
    plt.show()
