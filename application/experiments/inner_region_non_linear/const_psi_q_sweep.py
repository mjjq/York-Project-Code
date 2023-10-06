def const_psi_q_sweep():
    """
    Find and plot the maximum layer_width*Delta' value for a tearing mode as a
    function of the on-axis safety factor.

    Used to determine whether the constant-psi approximation remains true
    over a range of on-axis safety factor values.
    """
    m=2
    n=1
    lundquist_number = 1e8
    q_rs = m/n
    axis_qs = np.linspace(q_rs/q(0.0)-1e-2, q_rs/q(1.0)+1e-2, 100)

    solution_scale_factor = 1e-5

    times = np.linspace(0.0, 1e8, 1000)

    d_delta_maxs = []

    for axis_q in axis_qs:
        td_sol, tm0 = solve_time_dependent_system(
            m, n, lundquist_number, axis_q, solution_scale_factor, times
        )

        d_delta = td_sol.delta_primes * td_sol.w_t

        d_delta_maxs.append(max(d_delta))

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(axis_qs, d_delta_maxs, color='black')

    ax.set_xlabel(r"On-axis safety factor")
    ax.set_ylabel(r"Peak $w(t) \Delta'[w(t)]$")

    fig.tight_layout()
    #plt.show()
    savefig(
        f"nl_const_psi_approx_q_sweep_(m,n,A)=({m},{n},{solution_scale_factor})"
    )
    plt.show()
