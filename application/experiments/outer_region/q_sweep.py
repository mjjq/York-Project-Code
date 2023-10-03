from matplotlib import pyplot as plt

import imports
from tearing_mode_solver.outer_region_solver import *

def q_sweep():
    """
    Calculate growth rates of different modes as a function of the on-axis
    safety factor.
    """
    modes = [
        (2,1),
        (3,2)
    ]

    axis_q_plt = np.linspace(0.0, 100.0, 200)

    lundquist_number = 1e8

    fig, ax = plt.subplots(1, figsize=(4.5,3))
    ax.hlines(
        0.0, min(axis_q_plt), max(axis_q_plt), color='black', linestyle='--',
        label='Marginal stability'
    )

    min_q, max_q = 1000000.0, -1000000.0

    for m,n in modes:
        print(m, n)
        q_rs = m/n
        axis_q = np.linspace(q_rs/q(0.0), q_rs/q(1.0), 50)
        results = [growth_rate(m,n,lundquist_number, q) for q in axis_q]
        delta_ps, growth_rates = zip(*results)

        min_q = min(min_q, min(axis_q))
        max_q = max(max_q, max(axis_q))

        ax.plot(axis_q, growth_rates, label=f"(m, n)=({m}, {n})")

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom=-top)
    ax.set_xlim(left=min_q, right=max_q)

    ax.set_xlabel("On-axis safety factor")
    ax.set_ylabel(r"Normalised growth rate ($\gamma/\bar{\omega}_A$)")

    ax.legend(prop={'size': 7}, ncol=2)
    fig.tight_layout()
    ax.grid(which='both')
    ms, ns = zip(*modes)
    #savefig(
    #    f"q_vs_growth_rate_(m,n)_{min(ms)}-{max(ms)}_{min(ns)}-{max(ns)}"
    #)
    plt.show()

if __name__=='__main__':
    q_sweep()
