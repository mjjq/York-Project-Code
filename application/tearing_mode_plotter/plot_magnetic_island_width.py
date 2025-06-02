from matplotlib import pyplot as plt
import numpy as np

from tearing_mode_solver.helpers import TimeDependentSolution

def phase_plot(ts: TimeDependentSolution):
    fig, axs = plt.subplots(3, figsize=(5,6))
    ax_w, ax_dw_dt, ax_phase = axs

    ax_w.plot(ts.times, ts.w_t)
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("w/a")

    dw_vec = np.diff(ts.w_t)
    dt_vec = np.diff(ts.times)

    dwdt = dw_vec/dt_vec

    ax_dw_dt.plot(ts.times[:-1], dwdt)
    ax_dw_dt.set_yscale('log')
    ax_dw_dt.set_xlabel("Time (s)")
    ax_dw_dt.set_ylabel("dw/dt (a/s)")

    ax_phase.plot(ts.w_t[:-1], dwdt)
    ax_phase.set_xlabel("w/a")
    ax_phase.set_ylabel("dw/dt (a/s)")

    ax_w.set_yscale('log')

    fig.tight_layout()

    plt.show()