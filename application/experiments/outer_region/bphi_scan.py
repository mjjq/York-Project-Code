import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from copy import deepcopy

from tearing_mode_solver.outer_region_solver import solve_system
from tearing_mode_solver.loizu_delta_prime import delta_prime_loizu, calculate_coefficients
from tearing_mode_plotter.plot_outer_region import plot_outer_region_solution
from jorek_tools.quasi_linear_model.get_tm_parameters import get_parameters
from matplotlib import colors as cm

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
        help="Use JOREK postproc expressions as input"
    )
    parser.add_argument(
        '-q', '--qprofile-filename', type=str,
        help="Use JOREK qprofile as input"
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

    params = get_parameters(
        args.exprs_filename,
        args.qprofile_filename,
        args.poloidal_mode_number,
        args.toroidal_mode_number
    )

    b_phi_scale_factors = np.arange(0.8, 1.2, 0.05)
    delta_primes = []
    r_s_vals = []

    island_widths = np.linspace(0.0, 0.5, 50)

    for b_phi_sf in b_phi_scale_factors:
        rs, qs = zip(*params.q_profile)
        qs = b_phi_sf * np.array(qs)
        params_new = deepcopy(params)
        params_new.q_profile = list(zip(rs, qs))
        loizu_coefs = calculate_coefficients(params_new)
        delta_primes.append(delta_prime_loizu(island_widths, loizu_coefs))
        r_s_vals.append(loizu_coefs.r_s)



    fig_w, ax_w = plt.subplots(1, figsize=(6,3))
    ax_w.set_xlabel("w/a")
    ax_w.set_ylabel("$r_s \Delta'$")
    ax_w.hlines(
        0.0, 
        min(island_widths), max(island_widths), 
        color='black', linestyle='--', 
        label="Marginal stability"
    )
    ax_w.grid()

    for i, b_phi_sf in enumerate(b_phi_scale_factors):
        b_phi = params.B0 * b_phi_sf
        r_s = r_s_vals[i]
        dp = delta_primes[i]
        ax_w.plot(island_widths, dp, label=r'$B_\phi=$'f'{b_phi:.2g}T')

    ax_w.legend(bbox_to_anchor=(1.05, 1.0))
    fig_w.tight_layout()

    fig, ax_j = plt.subplots(1, figsize=(5,4))
    ax_j.grid()
    ax_j.set_xlabel("r/a")
    ax_j.set_ylabel("$J_\phi/J_{\phi,0}$")

    rs, js = zip(*params.j_profile)
    ax_j.plot(rs, js, color='black')
    for i, r_s in enumerate(r_s_vals):
        b_phi = params.B0 * b_phi_scale_factors[i]
        color = cm.to_hex(plt.cm.tab10(i))
        ax_j.vlines(
            r_s, min(js), max(js), label=r'$B_\phi=$'f'{b_phi:.2g}T',
            linestyle='--', color=color)

    ax_j.legend()
    fig.tight_layout()


    plt.show()
    return

if __name__=='__main__':
    ql_tm_vs_time()
