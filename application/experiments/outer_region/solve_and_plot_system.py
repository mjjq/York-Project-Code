from matplotlib import pyplot as plt

import imports
from tearing_mode_solver.outer_region_solver import *

def solve_and_plot_system():
    """
    Generate set of plots for an outer perturbed flux solution in addition to
    the q and current profiles of the plasma.
    """
    poloidal_mode = 3
    toroidal_mode = 2
    axis_q = 1.0

    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)

    delta_p = delta_prime(tm)

    print(f"Delta prime = {delta_p}")

    fig, axs = plt.subplots(3, figsize=(6,10), sharex=True)
    ax, ax2, ax3 = axs

    #ax4.plot(r_range_fwd, dpsi_dr_forwards)
    #ax4.plot(r_range_bkwd, dpsi_dr_backwards)

    ax.plot(
        tm.r_range_fwd, tm.psi_forwards, label='Solution below $\hat{r}_s$'
    )
    ax.plot(
        tm.r_range_bkwd, tm.psi_backwards, label='Solution above $\hat{r}_s$'
    )
    #rs_line = ax.vlines(
         #tm.r_s, ymin=0.0, ymax=np.max([tm.psi_forwards, tm.psi_backwards]),
         #linestyle='--', color='red',
         #label=f'Rational surface $\hat{{q}}(\hat{{r}}_s) = {poloidal_mode}/{toroidal_mode}$'
    #)

    ax3.set_xlabel("Normalised minor radial co-ordinate (r/a)")
    ax.set_ylabel("Normalised perturbed flux ($\delta \psi / a^2 J_\phi$)")

    r = np.linspace(0, 1.0, 100)
    ax2.plot(r, dj_dr(r))
    ax2.set_ylabel("Normalised current gradient $[d\hat{J}_\phi/d\hat{r}]$")
    ax2.vlines(
        tm.r_s,
        ymin=np.min(dj_dr(r)),
        ymax=np.max(dj_dr(r)),
        linestyle='--',
        color='red'
    )

    ax3.plot(r, np.vectorize(q)(r))
    ax3.set_ylabel("Normalised q-profile $[\hat{q}(\hat{r})]$")
    ax3.vlines(
        tm.r_s,
        ymin=np.min(q(r)),
        ymax=np.max(q(r)),
        linestyle='--',
        color='red'
    )
    #ax3.set_yscale('log')
    ax.legend(prop={'size': 8})

    plt.show()

if __name__=='__main__':
    solve_and_plot_system()
