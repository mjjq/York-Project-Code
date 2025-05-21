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


def plot_growth(times, dpsi_t, psi_t):
    fig_growth, ax_growth = plt.subplots(1, figsize=(4.5,3))

    ax_growth.plot(times, dpsi_t/psi_t, color='black')
    ax_growth.set_ylabel(r'Growth rate $\delta\dot{\psi}^{(1)}/\delta\psi^{(1)}$')
    ax_growth.set_xlabel(r'Normalised time $1/\bar{\omega}_A$')

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig_growth.tight_layout()

    ax_growth.set_xscale('log')
    #orig_fname, ext = os.path.splitext(os.path.basename(model_data_filename))
    #savefig(f"{orig_fname}_growth_rate")

def plot_outer_region_solution(params: TearingModeParameters, 
                               jorek_psi_data: Four2DProfile):
    tm = solve_system(params)

    dp = delta_prime(tm)
    print(f"r_s Delta' = {tm.r_s * dp}")
    s = magnetic_shear(params.q_profile, tm.r_s)
    print(f"Shear at r_s = {s}")
    print(f"r_s = {tm.r_s}")

    fig, axs = plt.subplots(2)
    ax, ax_dpsi_dr = axs
    
    max_psi = max(
        np.max(tm.psi_forwards), 
        np.max(tm.psi_backwards)
    )
    ax.plot(
        tm.r_range_fwd, tm.psi_forwards/max_psi, color='black', label='Model'
    )
    ax.plot(
        tm.r_range_bkwd, tm.psi_backwards/max_psi, color='black'
    )
    
   
    ax_dpsi_dr.plot(
        tm.r_range_fwd, tm.dpsi_dr_forwards/max_psi, color='black'
    )
    ax_dpsi_dr.plot(
        tm.r_range_bkwd, tm.dpsi_dr_backwards/max_psi, color='black'
    )
    
    
    jorek_rs = jorek_psi_data.r_minor/np.max(jorek_psi_data.r_minor)
    jorek_psi = jorek_psi_data.psi
    ax.plot(
        jorek_rs, jorek_psi/max(jorek_psi), color='red', 
        label='JOREK', linestyle='--'
    )
    
    jorek_psi_spline = UnivariateSpline(jorek_rs, jorek_psi, s=0.0)
    jorek_dpsi_dr = jorek_psi_spline.derivative()(jorek_rs)
    ax_dpsi_dr.plot(
        jorek_rs, jorek_dpsi_dr/max(jorek_psi), color='red', linestyle='--'
    )
    
    ax.set_ylabel("Normalised perturbed flux")
    ax_dpsi_dr.set_ylabel("Normalised $\partial \delta\psi^{(1)}/\partial r$") 
    ax_dpsi_dr.set_xlabel("Normalised minor radial co-ordinate (a)")
    
    fig.legend()

    savefig("outer_soln_comparison")


def test_energy_calculation(params: TearingModeParameters):
    tm = solve_system(params)

    norm_energy_int = normalised_energy_integral(tm, params)

    print("Normalised energy")
    print(norm_energy_int)

    psi_rs = 1e-10

    e = energy(psi_rs, params, norm_energy_int)

    print(f"Magnetic energy: {e}")
    

def ql_tm_vs_time(psi_current_prof_filename: str,
                  q_prof_filename: str,
                  jorek_fourier_filename: str,
                  poloidal_mode_number: int,
                  toroidal_mode_number: int):
    """
    Numerically solve the quasi-linear time-dependent tearing mode problem
    and plot the perturbed flux and layer width as functions of time.
    """
    
    params = get_parameters(
        psi_current_prof_filename,
        q_prof_filename,
        poloidal_mode_number,
        toroidal_mode_number
    )

    jorek_psi_data = read_four2d_profile_filter(
        jorek_fourier_filename,
        poloidal_mode_number
    )

    plot_outer_region_solution(params, jorek_psi_data)

    test_energy_calculation(params)

    
    
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
        '-exf', '--exprs-filename',
        type=str,
        help="Name of the exprs_averaged postproc filename",
        default="exprs_averaged_s00000.dat"
    )
    parser.add_argument(
        '-qf', '--qprof-filename',
        type=str,
        help="Name of the qprofile postproc filename",
        default="qprofile_s00000.dat"
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


    
    psi_current_prof_filename = args.exprs_filename
    q_prof_filename = args.qprof_filename
    jorek_fourier_filename = args.fourier_filename
    poloidal_mode_number = args.poloidal_mode_number
    toroidal_mode_number = args.toroidal_mode_number

    ql_tm_vs_time(
        psi_current_prof_filename,
        q_prof_filename,
        jorek_fourier_filename,
        poloidal_mode_number,
        toroidal_mode_number
    )
