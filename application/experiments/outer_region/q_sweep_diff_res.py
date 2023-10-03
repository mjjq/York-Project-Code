from matplotlib import pyplot as plt

import imports
from tearing_mode_solver.outer_region_solver import *


def q_sweep_diff_res():
    """
    Calculate growth rates of a tearing mode as a function of the on-axis
    safety factor for different tearing mode resolutions.
    """
    m,n = (3,2)
    resolutions = [1e-5, 1e-6, 1e-7]

    axis_q_plt = np.linspace(0.0, 100.0, 200)

    lundquist_number = 1e8

    fig, ax = plt.subplots(1, figsize=(4.5,3))
    ax.hlines(
        0.0, min(axis_q_plt), max(axis_q_plt), color='black', linestyle='--',
        label='Marginal stability'
    )

    min_q, max_q = 1000000.0, -1000000.0

    for res in resolutions:
        print(m, n)
        q_rs = m/n
        axis_q = np.linspace(q_rs/q(0.0), q_rs/q(1.0), 200)
        results = [
            growth_rate(m,n,lundquist_number, q, res) for q in axis_q
        ]
        delta_ps, growth_rates = zip(*results)

        min_q = min(min_q, min(axis_q))
        max_q = max(max_q, max(axis_q))

        growth_min = 0.9*np.array(growth_rates)
        growth_max = 1.1*np.array(growth_rates)

        ax.plot(
            axis_q,
            growth_rates,
            label=r"$\delta \hat{r}$"+f"={res}"
        )

        ax.fill_between(axis_q, growth_min, growth_max, alpha=0.4)

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom=-top)
    ax.set_xlim(left=min_q, right=max_q)

    ax.set_xlabel("On-axis safety factor")
    ax.set_ylabel(r"Normalised growth rate ($\gamma/\bar{\omega}_A$)")

    ax.legend(prop={'size': 7}, ncol=2)
    fig.tight_layout()
    ax.grid(which='both')
    savefig(
        f"q_sweep_res_test_(m,n)_{m},{n}"
    )
    plt.show()

if __name__=='__main__':
    q_sweep_diff_res()
