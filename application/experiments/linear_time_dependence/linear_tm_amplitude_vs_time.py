import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem

import imports

from tearing_mode_solver.time_dependent_solver import \
    solve_time_dependent_system

def linear_tm_amplitude_vs_time():
    """
    Plot amplitude of the flux at the resonant surface as a function of time
    for a linear tearing mode.
    """
    m=4
    n=2
    lundquist_number = 1e8

    res_f, res_b, tm, t_range = solve_time_dependent_system(
        m, n, lundquist_number, 1.0, np.linspace(0.0, 1e5, 100)
    )

    fig, ax = plt.subplots(1, figsize=(4,3))

    res_amplitudes = [psi_f[-1] for psi_f in res_f]

    ax.set_yscale('log')
    ax.scatter(t_range, res_amplitudes, s=1)

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(
        """Normalised perturbed flux at
        resonant surface [$\delta \hat{\psi}^{(1)}(r_s)$]"""
    )

    fig.tight_layout()

    plt.savefig(f"res_amplitude_vs_time_(m,n)={m},{n}.png", dpi=300)

    dt = t_range[-1] - t_range[-2]
    dpsi_dt = np.gradient(res_amplitudes, dt)
    growth_rate = dpsi_dt/res_amplitudes

    growth_rate_clipped = growth_rate[1:-2]
    print(f"""Average growth_rate = {np.mean(growth_rate_clipped)}
          +/- {sem(growth_rate_clipped)}""")

    fig2, ax2 = plt.subplots(1, figsize=(4,3))
    ax2.plot(t_range, growth_rate)

    ax2.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax2.set_ylabel(r"Normalised growth rate ($\gamma/\bar{\omega}_A$)")

    fig2.tight_layout()

if __name__=='__main__':
    linear_tm_amplitude_vs_time()
