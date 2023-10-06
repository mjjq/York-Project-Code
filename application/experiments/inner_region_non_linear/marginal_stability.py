import numpy as np
from scipy.stats import sem
from matplotlib import pyplot as plt

import imports

from tearing_mode_solver.outer_region_solver import (
    q
)
from tearing_mode_solver.nl_td_solver import solve_time_dependent_system
from tearing_mode_solver.helpers import savefig

def marginal_stability(poloidal_mode: int = 2, toroidal_mode: int = 2):
    """
    Solve time dependent NL equation for multiple q-values. Plot
    final island width as a function of q(0)
    """
    m=poloidal_mode
    n=toroidal_mode
    lundquist_number = 1e8
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e8, 100)

    q_rs = m/n
    axis_qs = np.linspace(q_rs/q(0.0)-1e-2, q_rs/q(1.0)+1e-2, 100)

    final_widths = []

    for axis_q in axis_qs:

        td0, tm0 = solve_time_dependent_system(
            m, n, lundquist_number, axis_q, solution_scale_factor, times
        )
        w_t = np.squeeze(td0.w_t)
        delta_primes = td0.delta_primes

        saturation_width = np.mean(w_t[-20:])
        saturation_width_sem = sem(w_t[-20:])
        final_widths.append(
            (saturation_width, saturation_width_sem, delta_primes[0])
        )

    fig, ax = plt.subplots(1, figsize=(4,3))
    ax2 = ax.twinx()

    means, sems, delta_primes = zip(*final_widths)
    means = np.array(means)
    sems = np.squeeze(np.array(sems))

    ax.set_title(f"(m, n)=({m}, {n})")

    ax2.plot(axis_qs, delta_primes, color='red', alpha=0.5)
    ax.plot(axis_qs, means, color='black')
    ax2.set_ylabel(r"$a\Delta'$ at $t=0$", color='red')
    #ax2.hlines(
        #0.0, min(axis_qs), max(axis_qs), color='red', linestyle='--'
    #)
    #ax.fill_between(axis_qs, means-sems, means+sems, alpha=0.3)

    # Set ylim for delta' plot so that zeros align
    ax.set_ylim(bottom=-0.01)
    ax_bottom, ax_top = ax.get_ylim()
    ax2_bottom, ax2_top = ax2.get_ylim()
    ax2_bottom_new = ax2_top * (ax_bottom/ax_top)
    ax2.set_ylim(bottom=ax2_bottom_new)

    ax.set_xlabel("On-axis safety factor")
    ax.set_ylabel("Saturated island width $(w/a)$")

    ax.grid(which='both')
    fig.tight_layout()

    savefig(f"q_sweep_nl_(m,n,A)=({m},{n},{solution_scale_factor})")

    plt.show()

def marg_stability_multi_mode():
    """
    Create marginal stability plots for multiple modes.
    """
    modes = [
        (2,1),
        (2,2),
        (2,3),
        (3,1),
        (3,2),
        (3,3)
    ]
    for m,n in modes:
        marginal_stability(m, n)


if __name__=='__main__':
    marginal_stability()
