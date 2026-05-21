from argparse import ArgumentParser
import numpy as np
from dataclasses import dataclass, fields
from typing import List
from matplotlib import pyplot as plt

from debug.log import logger

from chease_tools.dr_term_at_q import read_columns, CheaseColumns
from chease_tools.get_tm_parameters import get_parameters, ggj_term, bootstrap_term

from tearing_mode_solver.outer_region_solver import diffusion_width
from tearing_mode_solver.loizu_delta_prime import delta_prime_loizu, calculate_coefficients

from rdcon_tools.delta_gw import time_from_g_filename

@dataclass 
class DeltaCLTimeSeries:
    times: np.array
    delta_p_cl: np.array

@dataclass
class MeasuredIslandWidth:
    times: np.array
    w_measured: np.array


@dataclass
class MREContributions:
    # Array of times
    times: np.array
    # Array of measured island width as a function of time
    w_measured: np.array
    # Array of classical delta_prime at zero island width
    delta_p_cl: np.array
    # Array of classical delta prime with finite island width
    delta_p_cl_finite_island: np.array
    # Array of GGJ delta prime contributions
    delta_p_ggj: np.array
    # Array of bootstrap delta prime contributions
    delta_p_bs: np.array
    # Array of island diffusion width values
    w_d: np.array
    # Array of rational surface radii
    r_s: np.array
    # Array of resistivity values at q=2
    resistivity: np.array

    def write(self, filename: str):
        cols = []
        names = []
        for field in fields(MREContributions):
            cols.append(getattr(self, field.name))
            names.append(field.name)

        cols = np.array(cols).T

        header = " ".join(names)
        np.savetxt(filename, cols, header=header)

    
def read_mre_contributions(filename: str) -> MREContributions:
    cols = np.loadtxt(filename)

    mre = MREContributions(
        [],[],[],[],[],[],[],[],[]
    )

    for i,field in enumerate(fields(MREContributions)):
        setattr(mre, field.name, cols[:,i])

    return mre


def mre_contributions_from_chease(chease_cols_list: List[CheaseColumns],
                                  chease_times: np.array,
                                  poloidal_mode_number: int,
                                  toroidal_mode_number: int,
                                  w_measured: MeasuredIslandWidth,
                                  chi_perp_0: float,
                                  chi_par_0: float) -> MREContributions:
    """
    Calculate different contributions to modified Rutherford equation
    from a set of CHEASE equilibria, classical delta_prime measurements
    (either from RDCON or cylindrical code), and measured island
    widths
    """
    ret = MREContributions(
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([])
    )

    q_surf = float(poloidal_mode_number/toroidal_mode_number)

    for time, equil in zip(chease_times, chease_cols_list):
        w_at_time = np.interp(time, w_measured.times, w_measured.w_measured)

        r_s = np.interp(q_surf, equil.q, equil.r_avg)
        eps = np.interp(q_surf, equil.q, equil.eps)
        shear = np.interp(q_surf, equil.q, equil.shear)

        w_d = diffusion_width(
            chi_perp_0, chi_par_0,
            r_s, eps, toroidal_mode_number,
            shear
        )

        ggj_vals = ggj_term(
            w_at_time, 
            args.poloidal_mode_number,
            args.toroidal_mode_number,
            equil,
            w_d
        )

        bootstrap_vals = bootstrap_term(
            w_at_time,
            args.poloidal_mode_number,
            args.toroidal_mode_number,
            equil,
            w_d
        )

        params = get_parameters(
            equil,
            args.poloidal_mode_number,
            args.toroidal_mode_number
        )
        loizu_coefs = calculate_coefficients(params)
        delta_p_classical = delta_prime_loizu(
            w_at_time,
            loizu_coefs
        )

        ret.times = np.append(ret.times, time)
        ret.w_measured = np.append(ret.w_measured, w_at_time)
        ret.delta_p_cl = np.append(ret.delta_p_cl, loizu_coefs.delta_prime)
        ret.delta_p_cl_finite_island = np.append(ret.delta_p_cl_finite_island, delta_p_classical)
        ret.delta_p_ggj = np.append(ret.delta_p_ggj, ggj_vals)
        ret.delta_p_bs = np.append(ret.delta_p_bs, bootstrap_vals)
        ret.w_d = np.append(ret.w_d, w_d)
        ret.r_s = np.append(ret.r_s, r_s)
        ret.resistivity = np.append(ret.resistivity, 0.0)

    return ret


def plot_mre_contributions(mre: MREContributions):
    fig, ax = plt.subplots(1, figsize=(5,4))

    ax.plot(
        mre.times, mre.r_s*mre.delta_p_cl_finite_island, 
        label=r"$\Delta'_{CL}$",
        linestyle='--'
    )
    ax.plot(
        mre.times, mre.r_s*mre.delta_p_ggj,
        label=r"$\Delta'_{GGJ}$",
        linestyle='--'
    )
    ax.plot(
        mre.times, mre.r_s*mre.delta_p_bs,
        label=r"$\Delta'_{BS}$",
        linestyle='--'
    )
    
    sum_of_contribs = mre.r_s*(
        mre.delta_p_cl_finite_island+
        mre.delta_p_ggj+
        mre.delta_p_bs
    )

    ax.plot(
        mre.times, sum_of_contribs,
        label="$\Delta'$",
        color='black'
    )
    ax.legend()

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$r_s \Delta'$")


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
        "-d", "--mre-data-filename",
        type=str,
        help = "Path to MRE data. Overrides -c.",
        default=""
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

    if not args.mre_data_filename:
        col_list = [read_columns(f) for f in args.chease_cols_files]
        times = [time_from_g_filename(f) for f in args.chease_cols_files]

        w_measured = MeasuredIslandWidth(
            np.linspace(0.0, 1.0, 100),
            [0.05]*100
        )

        mre_vals = mre_contributions_from_chease(
            col_list,
            times,
            args.poloidal_mode_number,
            args.toroidal_mode_number,
            w_measured,
            args.chi_perp,
            args.chi_parallel
        )

        plot_mre_contributions(mre_vals)

        mre_vals.write("test_mre.txt")
    else:
        mre_vals = read_mre_contributions(args.mre_data_filename)

        plot_mre_contributions(mre_vals)

    plt.show()


    # j_bs_profile = MacroscopicQuantity(args.bootstrap_exprs_file)
    # j_bs_profile.load_x_values_by_index(0)
    # j_bs_profile.load_y_values_by_index(1)

    

    