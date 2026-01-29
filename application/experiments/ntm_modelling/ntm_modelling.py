from argparse import ArgumentParser
import numpy as np

from debug.log import logger

from chease_tools.dr_term_at_q import read_columns, CheaseColumns
from chease_tools.get_tm_parameters import get_parameters
from jorek_tools.macroscopic_vars_analysis.plot_quantities import MacroscopicQuantity

from tearing_mode_solver.bootstrap import ntm_bootstrap_term
from tearing_mode_solver.ggj import ntm_ggj_term
from tearing_mode_solver.loizu_delta_prime import delta_prime_loizu, calculate_coefficients

def ggj_term(w: float,
             poloidal_mode_number: float,
             toroidal_mode_number: float,
             chease_cols: CheaseColumns,
             w_d: float) -> float:
    q_s = poloidal_mode_number/toroidal_mode_number

    # Note: Chease outputs -d_r, so negate here
    d_r = -np.interp(q_s, chease_cols.q, chease_cols.d_r)
    beta_p = np.interp(q_s, chease_cols.q, chease_cols.beta_p)

    print(d_r)

    return ntm_ggj_term(w, d_r, w_d)

def bootstrap_term(w: float,
                   poloidal_mode_number: float,
                   toroidal_mode_number: float,
                   chease_cols: CheaseColumns,
                   w_d: float) -> float:
    q_s = poloidal_mode_number/toroidal_mode_number
    r_maj = chease_cols.r_avg[0]
    f_val = np.interp(
        q_s,
        chease_cols.q,
        chease_cols.F
    )
    shear_rs = np.interp(
        q_s,
        chease_cols.q,
        chease_cols.shear
    )

    # j_bs_rs = np.interp(
    #     psi_rs,
    #     bootstrap_profile.x_values,
    #     bootstrap_profile.y_values
    # )
    j_bs_rs = np.interp(
        q_s,
        chease_cols.q,
        chease_cols.j_bs
    )

    # Above is in chease units, but the function below
    # requires SI-units. Since the units of
    # R/B^2 * <j.B> cancel, only need to remove a factor
    # of mu0 from <j.B>
    mu0 = 4e-7 * np.pi
    j_bs_rs = j_bs_rs/mu0

    logger.debug(
        "r_maj, f_val, q_s, shear_rs, j_bs_rs", 
        r_maj, f_val, q_s, shear_rs, j_bs_rs
    )
    return ntm_bootstrap_term(
        w, r_maj, f_val, q_s, shear_rs, j_bs_rs, w_d
    )

if __name__=='__main__':
    logger.setLevel(1)
    parser = ArgumentParser()

    parser.add_argument(
        "chease_cols_file", 
        type=str, help="Path to chease_cols.out"
    )
    # parser.add_argument(
    #     "bootstrap_exprs_file", 
    #     type=str, help="Path to JOREK postproc bootstrap current file"
    # )
    parser.add_argument(
        "-m", "--poloidal-mode-number", type=int, default=2,
        help="Poloidal mode number"
    )
    parser.add_argument(
        "-n", "--toroidal-mode-number", type=int, default=1,
        help="Toroidal mode number"
    )
    parser.add_argument(
        '-wd', "--wd", type=float, default=0.0,
        help="Diffusion width (normalised to minor radius)"
    )
    parser.add_argument(
        '-s', "--scale-factor", type=float, default=1.0,
        help="Scale factor for q-profile (to simulate B_phi ramp)"
    )

    args = parser.parse_args()

    chease_cols = read_columns(args.chease_cols_file)
    chease_cols.q = args.scale_factor*chease_cols.q

    q_s = args.poloidal_mode_number/args.toroidal_mode_number
    r_s = np.interp(
        q_s,
        chease_cols.q,
        chease_cols.eps
    )

    w_vals = np.logspace(-3, np.log10(0.5), 100)

    ggj_vals = ggj_term(
        w_vals, 
        args.poloidal_mode_number,
        args.toroidal_mode_number,
        chease_cols,
        args.wd
    )

    bootstrap_vals = bootstrap_term(
        w_vals,
        args.poloidal_mode_number,
        args.toroidal_mode_number,
        chease_cols,
        args.wd
    )

    params = get_parameters(
        chease_cols,
        args.poloidal_mode_number,
        args.toroidal_mode_number
    )
    loizu_coefs = calculate_coefficients(params)
    delta_p_classical = delta_prime_loizu(
        w_vals,
        loizu_coefs
    )
    

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, figsize=(5,4))
    ax.set_title("$B_{\phi,0}/B_{\phi,0,exp}=$"f"{args.scale_factor}")
    ax.plot(w_vals, r_s*delta_p_classical, label="$r_s \Delta'_{CL}$", linestyle='--')
    ax.plot(w_vals, r_s*ggj_vals, label="$r_s \Delta'_{GGJ}$", linestyle='--')
    ax.plot(w_vals, r_s*bootstrap_vals, label="$r_s \Delta'_{BS}$", linestyle='--')
    ax.plot(
        w_vals, 
        r_s*(delta_p_classical+ggj_vals+bootstrap_vals), 
        label="$r_s \Delta'_{all}$",
        color='black'
    )
    ax.hlines(0.0, xmin=0.0, xmax=max(w_vals), color='black', linestyle='--')
    ax.set_xlabel("w/a")
    ax.set_ylabel("$r_s \Delta'(w)$")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()


    # j_bs_profile = MacroscopicQuantity(args.bootstrap_exprs_file)
    # j_bs_profile.load_x_values_by_index(0)
    # j_bs_profile.load_y_values_by_index(1)

    

    