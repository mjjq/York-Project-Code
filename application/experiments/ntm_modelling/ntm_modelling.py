from argparse import ArgumentParser
import numpy as np

from debug.log import logger

from chease_tools.dr_term_at_q import read_columns
from experiments.ntm_modelling.mre_time_series import mre_contributions_single

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
        "-xp", "--chi-perp", type=float, default=7e-8,
        help="On-axis perpendicular thermal diffusion coefficient"
    )
    parser.add_argument(
        "-xpa", "--chi-parallel", type=float, default=17.5,
        help="On-axis perpendicular thermal diffusion coefficient"
    )
    parser.add_argument(
        '-s', "--scale-factor", type=float, default=1.0,
        help="Scale factor for q-profile (to simulate B_phi ramp)"
    )

    args = parser.parse_args()

    chease_cols = read_columns(args.chease_cols_file)
    chease_cols.q = args.scale_factor*chease_cols.q

    w_vals = np.logspace(-3, np.log10(0.3), 100)

    mre = mre_contributions_single(
        w_vals, chease_cols,
        args.poloidal_mode_number,
        args.toroidal_mode_number,
        args.chi_perp,
        args.chi_parallel
    )
    r_s = mre.r_s
    delta_p_classical = mre.delta_p_cl_finite_island
    ggj_vals = mre.delta_p_ggj
    bootstrap_vals = mre.delta_p_bs
    

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, figsize=(5,4))
    ax.set_title("$B_{\phi,0}/B_{\phi,0,exp}=$"f"{args.scale_factor}")
    ax.plot(w_vals, r_s*delta_p_classical, label="Classical", linestyle='--')
    ax.plot(w_vals, r_s*ggj_vals, label="GGJ", linestyle='--')
    ax.plot(w_vals, r_s*bootstrap_vals, label="Bootstrap", linestyle='--')
    ax.plot(
        w_vals, 
        r_s*(delta_p_classical+ggj_vals+bootstrap_vals), 
        label="$r_s \Delta'_{all}$",
        color='black'
    )
    ax.hlines(0.0, xmin=0.0, xmax=max(w_vals), color='black', linestyle='--')
    ax.set_xlabel("w/a")
    ax.set_ylabel("$r_s \Delta'(w/a)$")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()


    # j_bs_profile = MacroscopicQuantity(args.bootstrap_exprs_file)
    # j_bs_profile.load_x_values_by_index(0)
    # j_bs_profile.load_y_values_by_index(1)

    

    