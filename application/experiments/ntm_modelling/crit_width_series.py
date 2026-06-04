from typing import Tuple, List
import numpy as np
from dataclasses import dataclass, fields
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from experiments.ntm_modelling.mre_time_series import MREContributions, mre_contributions_single
from chease_tools.dr_term_at_q import read_columns, CheaseColumns
from debug.log import logger
from rdcon_tools.delta_gw import time_from_g_filename

def find_zeros(x: np.array,
               y: np.array) -> np.array:
    """
    Find zeros or zero-crossings in x and y data
    """
    signs = np.sign(y)
    diff = np.abs(np.diff(signs))
    crossings = np.argwhere(diff > 0).flatten()
    print(crossings)

    zeros = []
    for crossing in crossings:
        x_tmp = x[crossing:crossing+2]
        y_tmp = y[crossing:crossing+2]
        # np.interp doesn't work if y isn't monotonically
        # increasing for some reason. So swap values
        # around if this is the case.
        if y_tmp[1] < y_tmp[0]:
            x_tmp = x_tmp[::-1]
            y_tmp = y_tmp[::-1]
        print(x_tmp, y_tmp)

        zero = np.interp(0.0, y_tmp, x_tmp)
        zeros.append(zero)

    return np.array(zeros)


def critical_island_widths(mre: MREContributions) -> Tuple[float, float]:
    """
    Calculate seed and saturation island widths from MRE output.
    Note: MRE output must contain w_vals covering entire minor radius
    e.g. np.linspace(1e-5, 1.0, 100.0)

    :return Tuple(w_seed, w_sat): If two zero-crossing for delta prime
        aren't found, then (None, None) is returned instead.
    """
    w_vals = mre.w_measured
    delta_p_total = mre.delta_p_cl_finite_island+mre.delta_p_ggj+mre.delta_p_bs

    zero_points = find_zeros(w_vals, delta_p_total)

    if len(zero_points)>=2:
        w_seed, w_sat = zero_points[:2]

        return (w_seed, w_sat)
    elif len(zero_points)==1:
        w_seed = zero_points[0]
        w_sat = 0.0
    else:
        print("No zero points!")
        w_seed = 0.0
        w_sat = 0.0

    return (w_seed, w_sat)

@dataclass
class CriticalWidthSeries:
    times: np.array
    w_seed: np.array
    w_sat: np.array

    def write(self, filename: str):
        cols = []
        names = []
        for field in fields(CriticalWidthSeries):
            cols.append(getattr(self, field.name))
            names.append(field.name)

        cols = np.array(cols).T

        header = " ".join(names)
        np.savetxt(filename, cols, header=header)

def crit_width_time_series(chease_cols_list: List[CheaseColumns],
                           chease_times: np.array,
                           poloidal_mode_number: int,
                           toroidal_mode_number: int,
                           chi_perp_0: float,
                           chi_par_0: float) -> CriticalWidthSeries:
    """
    Get critical island widths (seed and saturations widths)
    from a series of CHEASE equilibria.
    """
    w_vals = np.linspace(1e-5, 1.0, 100)

    w_seeds = []
    w_sats = []
    for time, equil in zip(chease_times, chease_cols_list):
        mre = mre_contributions_single(
            w_vals, equil, poloidal_mode_number, toroidal_mode_number,
            chi_perp_0, chi_par_0
        )
        w_seed, w_sat = critical_island_widths(mre)
        print(w_seed, w_sat)
        w_seeds.append(w_seed)
        w_sats.append(w_sat)

    return CriticalWidthSeries(
        times, np.array(w_seeds), np.array(w_sats)
    )

if __name__=='__main__':
    logger.setLevel(1)
    parser = ArgumentParser()

    parser.add_argument(
        "-c", "--chease-cols-files", 
        type=str, nargs='+', 
        help="Path to list of chease_cols.out "
        "Note: File or folder name must contain "
        "number recording time of the current equilibrium "
        "in float format "
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
        "-xp", "--chi-perp", type=float, default=7e-8,
        help="On-axis perpendicular thermal diffusion coefficient"
    )
    parser.add_argument(
        "-xpa", "--chi-parallel", type=float, default=17.5,
        help="On-axis perpendicular thermal diffusion coefficient"
    )
    args = parser.parse_args()

    col_list = [read_columns(f) for f in args.chease_cols_files]
    times = [time_from_g_filename(f) for f in args.chease_cols_files]

    crit_widths = crit_width_time_series(
        col_list,
        times,
        args.poloidal_mode_number,
        args.toroidal_mode_number,
        args.chi_perp,
        args.chi_parallel
    )
    crit_widths.write("crit_widths.txt")

    fig, ax = plt.subplots(2, sharex=True)
    ax_seed, ax_sat = ax
    ax[-1].set_xlabel('Time (s)')
    for ax_i in ax:
        ax_i.grid()

    ax_seed.plot(crit_widths.times, crit_widths.w_seed)
    ax_seed.set_ylabel("$w_{seed}/a$")

    ax_sat.plot(crit_widths.times, crit_widths.w_sat)
    ax_sat.set_ylabel("$w_{sat}/a$")

    fig.tight_layout()
    plt.show()