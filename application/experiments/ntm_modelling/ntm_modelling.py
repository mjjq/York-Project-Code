from argparse import ArgumentParser
import numpy as np

from debug.log import logger

from chease_tools.dr_term_at_q import read_columns, CheaseColumns
from chease_tools.get_tm_parameters import scale_profiles
from experiments.ntm_modelling.mre_time_series import (
    mre_contributions_single, read_measured_w_data, MeasuredIslandWidth
)
from experiments.ntm_modelling.compare_dw_dt import compare_dw_dt

def avg_island_width_to_outboard(chease_cols: CheaseColumns,
                                 w_measured: MeasuredIslandWidth,
                                 poloidal_mode_number: int,
                                 toroidal_mode_number: int) -> MeasuredIslandWidth:
    """
    Convert poloidally averaged island
    width to outboard island width

    :param chease_cols: CHEASE equilibrium data
    :param w_measured: Measured island width normalised to minor radius
    """
    if not w_measured.normalised:
        raise ValueError("Island width must be normalised!")
    
    q_s = float(poloidal_mode_number/toroidal_mode_number)

    rho_rs = np.interp(
        q_s,
        chease_cols.q,
        chease_cols.s
    )

    rho_max = rho_rs + 0.5*w_measured.w_measured
    rho_min = rho_rs - 0.5*w_measured.w_measured

    a_min = 0.5*(chease_cols.r_outboard[-1]+chease_cols.r_inboard[-1])

    R_min, R_max = np.interp(
        [rho_min, rho_max],
        chease_cols.s,
        chease_cols.r_outboard
    )

    w_out = (R_max - R_min)/a_min

    return MeasuredIslandWidth(
        w_measured.times,
        w_out,
        w_measured.w_measured_err,
        True
    )

        

if __name__=='__main__':
    logger.setLevel(1)
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
        "-xp", "--chi-perp", type=float, default=0.15499,
        help="On-axis perpendicular thermal diffusion coefficient"
    )
    parser.add_argument(
        "-xpa", "--chi-parallel", type=float, default=1.0934e7,
        help="On-axis perpendicular thermal diffusion coefficient"
    )
    parser.add_argument(
        "-e", "--resistivity", type=float, default=1.9413e-7,
        help="Resistivity at the rational surface"
    )
    parser.add_argument(
        '-s', "--scale-factor", type=float, default=1.0,
        help="Scale factor for q-profile (to simulate B_phi ramp)"
    )
    parser.add_argument(
        '-g', "--ggj-scale-factor", type=float, default=1.0,
        help="Scale factor for GGJ (to simulate rMHD)"
    )
    parser.add_argument(
        "-w", "--island-width-data-filename",
        type=str,
        help="Path to measured island width time trace.",
        default=""
    )

    args = parser.parse_args()

    chease_cols = read_columns(args.chease_cols_file)
    scale_profiles(chease_cols, args.scale_factor)

    w_vals = np.logspace(-3, np.log10(0.3), 100)

    mre_theory = mre_contributions_single(
        w_vals, chease_cols,
        args.poloidal_mode_number,
        args.toroidal_mode_number,
        args.chi_perp,
        args.chi_parallel
    )
    r_s = mre_theory.r_s 
    mu0 = 4e-7*np.pi
    eta = args.resistivity
    rutherford_scale = 1.22 # See Kleiner 2016, expression for f_n
    scale_fac = rutherford_scale*eta/mu0
    print(scale_fac)

    delta_p_classical = mre_theory.delta_p_cl_finite_island
    ggj_vals = args.ggj_scale_factor * mre_theory.delta_p_ggj
    bootstrap_vals = mre_theory.delta_p_bs

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, figsize=(4,3))
    #ax.set_title("$B_{\phi,0}/B_{\phi,0,exp}=$"f"{args.scale_factor}")
    ax.plot(w_vals, scale_fac*delta_p_classical, label="Classical", linestyle='--')
    ax.plot(w_vals, scale_fac*ggj_vals, label="GGJ", linestyle='--')
    ax.plot(w_vals, scale_fac*bootstrap_vals, label="Bootstrap", linestyle='--')
    ax.plot(
        w_vals, 
        scale_fac*(delta_p_classical+ggj_vals+bootstrap_vals), 
        label="Total",
        color='black'
    )
    ax.hlines(0.0, xmin=0.0, xmax=max(w_vals), color='black', linestyle='--')
    ax.set_xlabel("w/a")
    ax.set_ylabel("$d(w/a)/dt$ (/s)")
#    fig.tight_layout()
    
    
    if args.island_width_data_filename:
        measured_data = read_measured_w_data(args.island_width_data_filename)
        measured_data = avg_island_width_to_outboard(
           chease_cols,
            measured_data,
            args.poloidal_mode_number,
            args.toroidal_mode_number
        )
        # Allow simulation to equilibriate after a millisecond
        t_filter = measured_data.times > 1e-3
        times = measured_data.times[t_filter]
        w_vals = measured_data.w_measured[t_filter]
        dw_dt = np.diff(w_vals)/np.diff(times)
        measured_delta_prime = dw_dt*mu0/eta/rutherford_scale
        measured_rs_delta_prime = mre_theory.r_s * measured_delta_prime
        ax.plot(w_vals[:-1], dw_dt, label="JOREK")

        #mre_measured = mre_contributions_single(
        #    w_vals, chease_cols,
        #    args.poloidal_mode_number,
        #    args.toroidal_mode_number,
        #    args.chi_perp,
        #    args.chi_parallel
        #)
        #mre_measured.times = times
        #mre_measured.resistivity = args.resistivity
        #compare_dw_dt(mre_measured)    

    ax.legend(ncol=2)
    ax.grid()
    fig.tight_layout()
    fig.savefig(f"mre_bt{args.scale_factor}.pdf")
    plt.show()
    
