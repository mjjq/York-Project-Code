

def plot_full_delql():
    """
    Plot the full quasi-linear layer width as a function of (X, t) on a heatmap.
    """
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "./output/18-08-2023_16:41_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

    x_range = (-4, 4)

    delqls, times, xs = del_ql_full(ql_sol, m, n, S, s, r_s, x_range)

    fig, ax = plt.subplots(1, figsize=(4.3,4))
    ax.set_xlim(left=1e4, right=3e5)
    im = plt.imshow(
        delqls,
        extent=[min(times), max(times), min(xs), max(xs)],
        vmax=0.002
    )
    ax.set_aspect(0.5*(max(times)-min(times))/(max(xs)-min(xs)))
    fig.colorbar(im, fraction=0.046, pad=0.04)

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    ax.set_ylabel(r"X")
    fig.tight_layout()

    savefig(f"delta_heatmap_(m,n)=({m},{n})")

