import numpy as np
from argparse import ArgumentParser
from typing import List
from matplotlib import pyplot as plt

from chease_tools.dr_term_at_q import read_columns, CheaseColumns
from chease_tools.get_tm_parameters import scale_profiles
from experiments.ntm_modelling.mre_time_series import (
    mre_contributions_single, read_measured_w_data, MeasuredIslandWidth
)
from experiments.ntm_modelling.ntm_modelling import avg_island_width_to_outboard


def plot_dw_dt(measured_data_array: List[MeasuredIslandWidth],
               chease_cols_list: List[CheaseColumns],
               poloidal_mode_number: int,
               toroidal_mode_number: int,
               labels = None):
    fig, axs = plt.subplots(ncols=2, figsize=(6,2),sharey=True)
    ax, ax2 = axs
    fig2, ax_w = plt.subplots(figsize=(3,2))
    ax_w.grid()
    for ax_i in axs:
        ax_i.grid()
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("w/a")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("d(w/a)/dt")
    ax2.set_xlabel("w/a")

    for i, measured_data in enumerate(measured_data_array):
        chease_cols = chease_cols_list[i]
        if labels:
            label = labels[i]
        else:
            label=""
        measured_data = avg_island_width_to_outboard(
           chease_cols,
           measured_data,
           poloidal_mode_number,
           toroidal_mode_number
        )
        # Allow simulation to equilibriate after a millisecond
        t_filter = measured_data.times > 1e-3
        times = measured_data.times[t_filter]
        w_vals = measured_data.w_measured[t_filter]
        dw_dt = np.diff(w_vals)/np.diff(times)
        ax.plot(times[:-1], dw_dt, label=label)
        ax2.plot(w_vals[:-1], dw_dt, label=label)

        ax_w.plot(times, w_vals)

    if labels:
        ax2.legend(prop={'size':8})
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig("dw_dt_scan.pdf")
    fig2.savefig("w_scan.pdf")

if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument(
        "chease_cols_file", 
        type=str, help="Path to chease_cols.out"
    )
    parser.add_argument(
        "-m", "--poloidal-mode-number", type=int, default=2,
        help="Poloidal mode number"
    )
    parser.add_argument(
        "-n", "--toroidal-mode-number", type=int, default=1,
        help="Toroidal mode number"
    )
    parser.add_argument(
        '-s', "--scale-factors", type=float, nargs='+', default=[],
        help="Scale factor for q-profile (to simulate B_phi ramp)"
    )
    parser.add_argument(
        "-w", "--island-width-data-filename",
        type=str,
        nargs='+',
        help="Path to measured island width time trace.",
        default=""
    )

    args = parser.parse_args()

    chease_cols = read_columns(args.chease_cols_file)

    if not args.scale_factors:
        args.scale_factors = [1.0]*len(args.island_width_data_filename)

    if len(args.scale_factors) != len(args.island_width_data_filename):
        raise ValueError("Number of scale factor must equal number of w_measured!")

    chease_cols_list = [chease_cols for i in args.scale_factors]
    labels = []
    for c,scale in zip(chease_cols_list, args.scale_factors):
        scale_profiles(c, scale)
        labels.append(r"$B_t/B_{t,ref}=$"f"{scale:.2g}")

    w_measured_array = [read_measured_w_data(f) for f in args.island_width_data_filename]

    plot_dw_dt(
        w_measured_array,
        chease_cols_list,
        args.poloidal_mode_number,
        args.toroidal_mode_number,
        labels=labels
    )
    plt.show()
