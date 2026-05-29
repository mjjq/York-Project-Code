from argparse import ArgumentParser
import numpy as np
from dataclasses import dataclass, fields
from typing import List, Optional
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

def read_measured_delta_prime_data(filename: str) -> DeltaCLTimeSeries:
    times, dp_measured = np.loadtxt(filename).T

    return DeltaCLTimeSeries(
        times = times,
        delta_p_cl=dp_measured
    )

@dataclass
class MeasuredIslandWidth:
    # Time in [s]
    times: np.array
    # Measured island width [m] if normalised=False, else [a]
    w_measured: np.array
    # Error in measured island width [m] if normalised=False, else [a]
    w_measured_err: np.array
    # Whether the island widths are normalised to minor radius
    normalised: bool

    def write(self, filename: str):
        cols = np.array([
            self.times, self.w_measured, self.w_measured_err
        ]).T

        header = f"times w_measured w_measured_err normalised {self.normalised}"
        np.savetxt(filename, cols, header=header)

def read_measured_w_data(filename: str) -> MeasuredIslandWidth:
    times, w_measured, w_measured_err = np.loadtxt(filename).T

    with open(filename, 'r') as f:
        header = f.readlines(1)[0]
        if "normalised True" in header:
            normalised = True
        elif "normalised False" in header:
            normalised = False
        else:
            print(
                "Warning: Unable to determine if island"
                "width data is normalised. Assume it is not."
            )
            normalised = False


    return MeasuredIslandWidth(
        times = times,
        w_measured= w_measured,
        w_measured_err=w_measured_err,
        normalised=normalised
    )


@dataclass
class MREContributions:
    # Array of times
    times: np.array
    # Array of measured island width as a function of time
    # units of [m]
    w_measured: np.array
    # Array of errors in measured island width as a function
    # of time units of [m]
    w_measured_err: np.array
    # Array of classical delta_prime at zero island width
    delta_p_cl: np.array
    # Array of classical delta prime with finite island width
    delta_p_cl_finite_island: np.array
    # Array of error in classical delta prime
    delta_p_cl_finite_island_err: np.array
    # Array of GGJ delta prime contributions
    delta_p_ggj: np.array
    # Array of GGJ delta prime errors
    delta_p_ggj_err: np.array
    # Array of bootstrap delta prime contributions
    delta_p_bs: np.array
    # Array of bootstrap delta prime errors
    delta_p_bs_err: np.array
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
        [],[],[],[],[],[],[],[],[],[],[],[],[]
    )

    for i,field in enumerate(fields(MREContributions)):
        setattr(mre, field.name, cols[:,i])

    return mre

def mre_contributions_single(w_vals: np.array,
                             equil: CheaseColumns,
                             poloidal_mode_number: int,
                             toroidal_mode_number: int,
                             chi_perp_0: float,
                             chi_par_0: float,
                             calc_delta_p_cl: bool = True) -> MREContributions:
    """
    Calculate all MRE contributions for a given CHEASE equilibrium

    Assume w_vals is normalised, i.e. in units of minor radius
    """
    q_surf = float(poloidal_mode_number/toroidal_mode_number)

    R_in_rs = np.interp(q_surf, equil.q, equil.r_inboard)
    R_out_rs = np.interp(q_surf, equil.q, equil.r_outboard)
    eps_rs = (R_out_rs-R_in_rs)/(R_out_rs+R_in_rs)

    R_in_max = equil.r_inboard[-1]
    R_out_max = equil.r_outboard[-1]
    eps_max = (R_out_max-R_in_max)/(R_out_max+R_in_max)

    w=w_vals

    # Calculate r_s in units of minor radius, not units of chease!
    r_s = eps_rs/eps_max
    shear = np.interp(q_surf, equil.q, equil.shear)

    w_d = diffusion_width(
        chi_perp_0, chi_par_0,
        r_s, 1.0/eps_max, toroidal_mode_number,
        shear
    )

    ggj_vals = ggj_term(
        w, 
        poloidal_mode_number,
        toroidal_mode_number,
        equil,
        w_d
    )

    bootstrap_vals = bootstrap_term(
        w,
        poloidal_mode_number,
        toroidal_mode_number,
        equil,
        w_d
    )

    if calc_delta_p_cl:
        try:
            params = get_parameters(
                equil,
                poloidal_mode_number,
                toroidal_mode_number
            )
            loizu_coefs = calculate_coefficients(params)
            # For now, don't evaluate at finite width
            delta_p_classical_finite_w = delta_prime_loizu(
                w,
                loizu_coefs
            )
            delta_p_classical = loizu_coefs.delta_prime
        except ValueError as e:
            print(f"Could not calculate delta prime")
            delta_p_classical = 0.0
            delta_p_classical_finite_w = [0.0]*len(w)
    else:
        delta_p_classical = 0.0
        delta_p_classical_finite_w = [0.0]*len(w)

    ret = MREContributions(
        None,
        w,
        None,
        delta_p_classical,
        delta_p_classical_finite_w,
        None,
        ggj_vals,
        None,
        bootstrap_vals,
        None,
        w_d,
        r_s,
        None
    )

    return ret

def mre_contributions_from_chease(chease_cols_list: List[CheaseColumns],
                                  chease_times: np.array,
                                  poloidal_mode_number: int,
                                  toroidal_mode_number: int,
                                  chi_perp_0: float,
                                  chi_par_0: float,
                                  w_measured: MeasuredIslandWidth,
                                  delta_p_cl: Optional[DeltaCLTimeSeries] = None,
                                  r0exp_chease: float = 0.8) -> MREContributions:
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
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([])
    )

    for time, equil in zip(chease_times, chease_cols_list):
        R_in_max = equil.r_inboard[-1]
        R_out_max = equil.r_outboard[-1]
        eps_max = (R_out_max-R_in_max)/(R_out_max+R_in_max)

        R0_geom_chease = 1.0
        a_chease = eps_max * R0_geom_chease
        a_si = a_chease*r0exp_chease
        # Island width given in units of metres. Convert to w/a
        # for consistency with units in this code.
        w_at_time = np.interp(
            time, w_measured.times, w_measured.w_measured
        )
        w_err_at_time = np.interp(
            time, w_measured.times, w_measured.w_measured_err
        )
        if not w_measured.normalised:
            w_at_time = w_at_time/a_si
            w_err_at_time = w_err_at_time/a_si

        w_min = w_at_time-w_err_at_time
        w_max = w_at_time+w_err_at_time

        # Evaluate for average width, and error bounds
        w=np.array([w_at_time, w_min, w_max])

        calc_cylindrical_delta_p = True
        if delta_p_cl:
            calc_cylindrical_delta_p = False

        mre_contribs = mre_contributions_single(
            w, equil, poloidal_mode_number, toroidal_mode_number,
            chi_perp_0, chi_par_0, calc_cylindrical_delta_p
        )
        ggj_avg, ggj_min, ggj_max = mre_contribs.delta_p_ggj
        ggj_err = (ggj_max-ggj_min)/2.0

        bs_avg, bs_min, bs_max = mre_contribs.delta_p_bs
        bs_err = (bs_max-bs_min)/2.0

        if delta_p_cl:
            # RDCON gives delta' in units of [/m]
            # Convert to units of aDelta' by multiplying
            # by minor radius
            delta_p_classical = a_si*np.interp(
                time,
                delta_p_cl.times,
                delta_p_cl.delta_p_cl
            )
            delta_p_classical_finite_w = a_si*delta_p_classical
            delta_pw_avg = delta_p_classical_finite_w
            delta_pw_err = 0.0
        else:
            delta_p_classical = mre_contribs.delta_p_cl
            delta_pw_avg, delta_pw_min, delta_pw_max = mre_contribs.delta_p_cl_finite_island
            delta_pw_err = 0.5*(delta_pw_max-delta_pw_min)

        w_at_time_si = a_si*w_at_time
        w_err_at_time_si = a_si*w_err_at_time

        ret.times = np.append(ret.times, time)
        ret.w_measured = np.append(ret.w_measured, w_at_time_si)
        ret.w_measured_err = np.append(ret.w_measured_err, w_err_at_time_si)
        ret.delta_p_cl = np.append(ret.delta_p_cl, delta_p_classical)
        ret.delta_p_cl_finite_island = np.append(
            ret.delta_p_cl_finite_island, delta_pw_avg
        )
        ret.delta_p_cl_finite_island_err = np.append(
            ret.delta_p_cl_finite_island_err, delta_pw_err
        )
        ret.delta_p_ggj = np.append(ret.delta_p_ggj, ggj_avg)
        ret.delta_p_ggj_err = np.append(ret.delta_p_ggj_err, ggj_err)
        ret.delta_p_bs = np.append(ret.delta_p_bs, bs_avg)
        ret.delta_p_bs_err = np.append(ret.delta_p_bs_err, bs_err)
        ret.w_d = np.append(ret.w_d, mre_contribs.w_d)
        ret.r_s = np.append(ret.r_s, mre_contribs.r_s)
        ret.resistivity = np.append(ret.resistivity, 0.0)

    return ret


def plot_mre_contributions(mre: MREContributions):
    fig, axs = plt.subplots(2, figsize=(5,5), sharex=True)
    ax, ax2 = axs

    ax.plot(
        mre.times, mre.r_s*mre.delta_p_cl_finite_island, 
        label=r"Classical",
        linestyle='--'
    )
    ax.fill_between(
        mre.times, 
        mre.r_s*(mre.delta_p_cl_finite_island-mre.delta_p_cl_finite_island_err),
        mre.r_s*(mre.delta_p_cl_finite_island+mre.delta_p_cl_finite_island_err),
        alpha=0.3,
        color='tab:blue'
    )
    ax.plot(
        mre.times, mre.r_s*mre.delta_p_ggj,
        label=r"GGJ",
        linestyle='--',
        color='tab:orange'
    )
    ax.fill_between(
        mre.times,
        mre.r_s*(mre.delta_p_ggj-mre.delta_p_ggj_err),
        mre.r_s*(mre.delta_p_ggj+mre.delta_p_ggj_err),
        alpha=0.3,
        color='tab:orange'
    )
    ax.plot(
        mre.times, mre.r_s*mre.delta_p_bs,
        label=r"Bootstrap",
        linestyle='--',
        color='tab:green'
    )
    ax.fill_between(
        mre.times,
        mre.r_s*(mre.delta_p_bs-mre.delta_p_bs_err),
        mre.r_s*(mre.delta_p_bs+mre.delta_p_bs_err),
        alpha=0.3,
        color='tab:green'
    )
    
    sum_of_contribs = mre.r_s*(
        mre.delta_p_cl_finite_island+
        mre.delta_p_ggj+
        mre.delta_p_bs
    )
    sum_err = mre.r_s*np.sqrt(
        mre.delta_p_cl_finite_island_err**2+
        mre.delta_p_bs_err**2+
        mre.delta_p_ggj_err**2
    )

    ax.plot(
        mre.times, sum_of_contribs,
        label="Total",
        color='black'
    )
    ax.fill_between(
        mre.times,
        sum_of_contribs-sum_err,
        sum_of_contribs+sum_err,
        alpha=0.3,
        color='black'
    )
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, 1.1),ncol=4,prop={'size': 8})

    #ax.set_xlabel("Time (s)")
    ax.set_ylabel("$r_s \Delta'$")

    #fig2, ax2 = plt.subplots(1, figsize=(5,4))
    ax2.plot(mre.times, 100.0*mre.w_measured)
    ax2.fill_between(
        mre.times, 
        100.0*(mre.w_measured-mre.w_measured_err),
        100.0*(mre.w_measured+mre.w_measured_err),
        alpha=0.5
    )
    ax2.hlines(
        2.0, 
        min(mre.times), max(mre.times), 
        linestyle='--',
        color='black',
        label='Noise threshold'
    )
    ax2.set_ylabel("Measured island width (cm)")
    ax2.legend()

    axs[-1].set_xlabel("Time (s)")
    for ax_l in axs:
        ax_l.grid()

    fig.tight_layout()

    figwd, axwd = plt.subplots(1)
    axwd.plot(mre.times, mre.w_d, label='Measured width')
    

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
        "-w", "--island-width-data-filename",
        type=str,
        help="Path to measured island width time trace.",
        default=""
    )
    parser.add_argument(
        "-rd", "--rdcon-data-filename",
        type=str,
        help="Path to RDCON island width data vs time",
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
    parser.add_argument(
        '-s', '--stationary-equilibrium', action='store_true',
        help="If activated, MRE is evaluated for"
        " the first equilibrium in chease_cols_files for the "
        "duration of the island width evolution"
    )

    args = parser.parse_args()

    if not args.mre_data_filename:
        if args.island_width_data_filename:
            w_measured = read_measured_w_data(
                args.island_width_data_filename
            )
        else:
            print("No island width data supplied, using w=0.05.")
            w_measured = MeasuredIslandWidth(
                np.linspace(0.0, 1.0, 100),
                [0.05]*100
            )

        col_list = [read_columns(f) for f in args.chease_cols_files]
        times = [time_from_g_filename(f) for f in args.chease_cols_files]

        if args.stationary_equilibrium:
            times = w_measured.times
            col_list = [col_list[0]]*len(times)

        if args.rdcon_data_filename:
            deltap_data = read_measured_delta_prime_data(
                args.rdcon_data_filename
            )
        else:
            deltap_data = None

        mre_vals = mre_contributions_from_chease(
            col_list,
            times,
            args.poloidal_mode_number,
            args.toroidal_mode_number,
            args.chi_perp,
            args.chi_parallel,
            w_measured,
            deltap_data
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

    

    
