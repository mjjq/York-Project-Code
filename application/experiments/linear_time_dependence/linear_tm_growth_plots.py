from matplotlib import pyplot as plt
import numpy as np

import imports
from tearing_mode_solver.time_dependent_solver import \
    solve_time_dependent_system

def linear_tm_growth_plots():
    """
    Plot the full outer solution as a function of minor radius at different
    times using the linear time-dependent solver.
    """
    m=3
    n=2
    lundquist_number = 1e8

    times = np.linspace(0.0, 1e4, 3)

    res_f, res_b, tm, t_range = solve_time_dependent_system(
        m, n, lundquist_number,1.0, times
    )

    fig, ax = plt.subplots(1, figsize=(4,3))

    for i, psi_f in enumerate(res_f):
        psi_b = res_b[i]

        psi = np.concatenate((psi_f, psi_b[::-1]))
        r = np.concatenate((tm.r_range_fwd, tm.r_range_bkwd[::-1]))

        ax.plot(r, psi, label=r'$\bar{\omega}_A t$='+f'{times[i]:.1e}')

    ax.vlines(
        tm.r_s, 0.0, np.max((np.max(res_f), np.max(res_b))), color='red', linestyle='--',
        label='$\hat{r}_s$='+f'{tm.r_s:.2f}'
    )

    ax.set_xlabel("Normalised minor radial co-ordinate $\hat{r}$")
    ax.set_ylabel("Normalised perturbed flux $\delta \hat{\psi}^{(1)}$")
    ax.legend(prop={'size':8})
    fig.tight_layout()

    plt.savefig(f"linear_tm_time_evo_(m,n)={m},{n}.png", dpi=300)


if __name__=='__main__':
    linear_tm_growth_plots()
