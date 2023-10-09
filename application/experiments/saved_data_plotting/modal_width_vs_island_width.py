import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

import imports
from tearing_mode_solver.outer_region_solver import (
    rational_surface, magnetic_shear
)
from tearing_mode_solver.delta_model_solver import nu, mode_width
from tearing_mode_solver.helpers import (
    classFromArgs, TimeDependentSolution, savefig
)
from tearing_mode_solver.outer_region_solver import island_width


def ql_modal_width_and_island_width():
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname_new = "./output/29-08-2023_10:53_new_ql_tm_time_evo_(m,n,A,q0)=(2,1,1e-10,1.0).csv"
    df_new = pd.read_csv(fname_new)
    ql_sol = classFromArgs(TimeDependentSolution, df_new)

    island_widths = island_width(
        ql_sol.psi_t,
        r_s,
        m,
        n,
        s
    )

    modal_widths = mode_width(
        ql_sol.psi_t,
        ql_sol.dpsi_dt,
        ql_sol.d2psi_dt2,
        r_s,
        m,
        n,
        s,
        S
    )
    modal_widths = modal_widths*2**(9/4) * r_s

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(ql_sol.times, island_widths,
        label=r'Magnetic island width $[w(t)]$',
        color='black'
    )
    ax.plot(ql_sol.times, modal_widths,
        label=r'Modal width $[2^{9/4}r_s \bar{\delta}_{ql}(t)]$',
        color='red', linestyle='--'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(prop={'size':7})

    ax.set_xlabel(r"Normalised time ($1/\bar{\omega}_A$)")
    ax.set_ylabel(r"Width ($a$)")

    fig.tight_layout()

    orig_fname, ext = os.path.splitext(os.path.basename(fname_new))
    savefig(f"width_comparison_{orig_fname}")

    plt.show()

if __name__=='__main__':
    ql_modal_width_and_island_width()
