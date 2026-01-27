from argparse import ArgumentParser
import numpy as np

from chease_tools.dr_term_at_q import read_columns, CheaseColumns
from jorek_tools.macroscopic_vars_analysis.plot_quantities import MacroscopicQuantity

from tearing_mode_solver.bootstrap import ntm_bootstrap_term
from tearing_mode_solver.ggj import ntm_ggj_term
from tearing_mode_solver.loizu_delta_prime import delta_prime_loizu

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

    return ntm_ggj_term(w, d_r, beta_p, w_d)

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

    return ntm_bootstrap_term(
        w, r_maj, f_val, q_s, shear_rs, j_bs_rs, w_d
    )

if __name__=='__main__':
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

    args = parser.parse_args()

    chease_cols = read_columns(args.chease_cols_file)

    w_vals = np.logspace(-3, -1, 100)

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

    from matplotlib import pyplot as plt
    plt.plot(w_vals, ggj_vals)
    plt.plot(w_vals, bootstrap_vals)
    plt.plot(w_vals, ggj_vals+bootstrap_vals)
    plt.grid()
    plt.show()


    # j_bs_profile = MacroscopicQuantity(args.bootstrap_exprs_file)
    # j_bs_profile.load_x_values_by_index(0)
    # j_bs_profile.load_y_values_by_index(1)

    

    