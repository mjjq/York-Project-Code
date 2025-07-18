import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from argparse import ArgumentParser

from jorek_tools.jorek_dat_to_array import (
    read_four2d_profile_filter, Four2DProfile
)
from tearing_mode_solver.outer_region_solver import (
    solve_system, normalised_energy_integral, energy, delta_prime, magnetic_shear
)
from tearing_mode_solver.helpers import (
    savefig, 
    TearingModeParameters
)
from jorek_tools.quasi_linear_model.get_tm_parameters import get_parameters


def plot_outer_region_solution(jorek_psi_data: Four2DProfile):
    fig, axs = plt.subplots(2)
    ax, ax_dpsi_dr = axs

    jorek_rs = jorek_psi_data.r_minor
    jorek_psi = jorek_psi_data.psi
    ax.plot(
        jorek_rs, jorek_psi/max(jorek_psi), 'x', color='black', 
        label='JOREK'
    )
    
    jorek_psi_spline = UnivariateSpline(jorek_rs, jorek_psi, s=0.0)
    jorek_dpsi_dr = jorek_psi_spline.derivative()(jorek_rs)
    ax_dpsi_dr.plot(
        jorek_rs, jorek_dpsi_dr/max(jorek_psi), 'x', color='black'
    )
    
    ax.set_ylabel("Normalised perturbed flux")
    ax_dpsi_dr.set_ylabel("Normalised $\partial \delta\psi^{(1)}/\partial r$") 
    ax_dpsi_dr.set_xlabel("Normalised minor radial co-ordinate (a)")
    
    fig.legend()

    savefig("psi_profile")

   

def ql_tm_vs_time(jorek_fourier_filename: str,
                  poloidal_mode_number: int,
                  toroidal_mode_number: int):
    jorek_psi_data = read_four2d_profile_filter(
        jorek_fourier_filename,
        poloidal_mode_number
    )

    plot_outer_region_solution(jorek_psi_data)

    plt.show()


if __name__=='__main__':
    parser = ArgumentParser(
        description="Plot outer region solution to delta psi for a given "\
        "tearing mode."
    )
    parser.add_argument(
        "fourier_filename", 
        type=str,
        help="Name of postproc fourier filename"
    )
    parser.add_argument(
        '-m', '--poloidal-mode-number',
        type=int,
        help="Poloidal mode number",
        default=2
    )
    parser.add_argument(
        '-n', '--toroidal-mode-number',
        type=int,
        help="toroidal mode number",
        default=1
    )
    args = parser.parse_args()

    jorek_fourier_filename = args.fourier_filename
    poloidal_mode_number = args.poloidal_mode_number
    toroidal_mode_number = args.toroidal_mode_number

    ql_tm_vs_time(
        jorek_fourier_filename,
        poloidal_mode_number,
        toroidal_mode_number
    )
