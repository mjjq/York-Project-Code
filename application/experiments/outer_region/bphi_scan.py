import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from copy import deepcopy

from tearing_mode_solver.outer_region_solver import solve_system
from tearing_mode_solver.loizu_delta_prime import delta_prime_loizu, calculate_coefficients
from tearing_mode_plotter.plot_outer_region import plot_outer_region_solution
from jorek_tools.quasi_linear_model.get_tm_parameters import get_parameters
from chease_tools.get_tm_parameters import get_parameters as chease_params
from chease_tools.dr_term_at_q import read_columns
from matplotlib import colors, cm

def ql_tm_vs_time():
    """
    Numerically solve the quasi-linear time-dependent tearing mode problem
    and plot the perturbed flux and layer width as functions of time.
    """
    parser = ArgumentParser(
        description="Plot outer region solution with debugging"
    )
    parser.add_argument(
        '-ex', '--exprs-filename', type=str,
        help="Use JOREK postproc expressions as input",
        default=""
    )
    parser.add_argument(
        '-q', '--qprofile-filename', type=str,
        help="Use JOREK qprofile as input",
        default=""
    )
    parser.add_argument(
        '-c', '--chease-filename', type=str,
        help="Use CHEASE columns as input",
        default=""
    )
    parser.add_argument(
        '-m', '--poloidal-mode-number',
        help="Poloidal mode number to analyse",
        type=int,
        default=2
    )
    parser.add_argument(
        '-n', '--toroidal-mode-number',
        help="Toroidal mode number to analyse",
        type=int,
        default=1
    )
    args=parser.parse_args()

    if args.exprs_filename and args.qprofile_filename:
        params = get_parameters(
            args.exprs_filename,
            args.qprofile_filename,
            args.poloidal_mode_number,
            args.toroidal_mode_number
        )
    elif args.chease_filename:
        cols = read_columns(args.chease_filename)
        params = chease_params(
            cols,
            args.poloidal_mode_number,
            args.toroidal_mode_number
        )
    else:
        print("Must specify either JOREK or CHEASE inputs! Exiting...")
        exit()

    b_phi_scale_factors = np.arange(0.8, 1.2, 0.01)
    delta_primes = []
    r_s_vals = []

    island_widths = np.linspace(0.0, 0.5, 50)
    island_widths[0] = 1e-5

    for b_phi_sf in b_phi_scale_factors:
        rs, qs = zip(*params.q_profile)
        qs = b_phi_sf * np.array(qs)
        params_new = deepcopy(params)
        params_new.q_profile = list(zip(rs, qs))
        loizu_coefs = calculate_coefficients(params_new)
        delta_primes.append(delta_prime_loizu(island_widths, loizu_coefs))
        r_s_vals.append(loizu_coefs.r_s)



    fig_w, ax_w = plt.subplots(1, figsize=(6,5))
    ax_w.set_xlabel("w/a")
    ax_w.set_ylabel("$B_{\phi,0}/B_{\phi,0,exp}$")

    min_dp = np.nanmin(np.array(delta_primes))
    max_dp = np.nanmax(np.array(delta_primes))
    if np.abs(max_dp) > np.abs(min_dp):
        min_dp = -max_dp
    else:
        max_dp = -min_dp

    b_phi = b_phi_scale_factors
    im = ax_w.imshow(
        delta_primes, 
        cmap='coolwarm',
        extent=[min(island_widths), max(island_widths), min(b_phi), max(b_phi)],
        aspect='auto',
        vmin=min_dp,
        vmax=max_dp,
        origin='lower'
    )
    cbar = plt.colorbar(im, ax=ax_w)
    cbar.set_label(r"$r_s\Delta'$")

    fig_w.tight_layout()

    fig, ax_j = plt.subplots(1, figsize=(5,4))
    ax_j.grid()
    ax_j.set_xlabel("r/a")
    ax_j.set_ylabel("$J_\phi/J_{\phi,0}$")

    rs, js = zip(*params.j_profile)
    ax_j.plot(rs, js, color='black')
    for i, r_s in enumerate(r_s_vals):
        b_phi = params.B0 * b_phi_scale_factors[i]
        color = colors.to_hex(plt.cm.tab10(i))
        ax_j.vlines(
            r_s, min(js), max(js), label=r'$B_\phi=$'f'{b_phi:.2g}T',
            linestyle='--', color=color)

    ax_j.legend()
    fig.tight_layout()


    plt.show()
    return

if __name__=='__main__':
    ql_tm_vs_time()
