import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from argparse import ArgumentParser
import glob
from typing import List, Tuple

from jorek_tools.h5_extractor import get_si_timesteps
from jorek_tools.jorek_dat_to_array import (
    read_four2d_profile_filter, Four2DProfile, read_four2d_profile,
    read_r_minor
)
from tearing_mode_solver.delta_model_solver import (
    island_width,
    TimeDependentSolution
)
from tearing_mode_solver.outer_region_solver import (
    rational_surface, magnetic_shear
)
from tearing_mode_solver.helpers import (
    savefig, 
    TearingModeParameters
)
from jorek_tools.quasi_linear_model.get_tm_parameters import get_parameters
from tearing_mode_plotter.plot_magnetic_island_width import phase_plot
from tearing_mode_solver.helpers import sim_to_disk

def island_width_from_jorek(psi_current_prof_filename: str,
                            q_prof_filename: str,
                            jorek_fourier_filenames: List[str],
                            jorek_h5_filenames: List[str],
                            poloidal_mode_number: int,
                            toroidal_mode_number: int) -> \
                                Tuple[TearingModeParameters, TimeDependentSolution]:
    params = get_parameters(
        psi_current_prof_filename,
        q_prof_filename,
        poloidal_mode_number,
        toroidal_mode_number
    )

    jorek_psi_data = [read_four2d_profile_filter(
        fname,
        poloidal_mode_number
    ) for fname in jorek_fourier_filenames]

    r_s = rational_surface(
        params.q_profile, poloidal_mode_number/toroidal_mode_number
    )
    r_minor = read_r_minor(psi_current_prof_filename)
    r_s_si = r_s*r_minor

    psi_rs = [np.interp(r_s_si, d.r_minor, d.psi) for d in jorek_psi_data]

    # Perform normalisation. See \autoref{eq:psi-j2ql} in lab book.
    psi_rs_normalised = list(np.array(psi_rs) / (params.B0 * r_minor**2))

    shear = magnetic_shear(params.q_profile, r_s)

    island_widths = island_width(
        psi_rs_normalised, r_s, 
        poloidal_mode_number, toroidal_mode_number, 
        shear
    )

    time_data = get_si_timesteps(jorek_h5_filenames)

    return params, TimeDependentSolution(
        time_data.t_si_vals,
        psi_rs,
        dpsi_dt=None,
        d2psi_dt2=None,
        w_t=island_widths,
        delta_primes=None
    )


if __name__=='__main__':
    parser = ArgumentParser(
        description="Plot outer region solution to delta psi for a given "\
        "tearing mode."
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
        '-hf', '--h5-filename-pattern',
        type=str,
        help="List of .h5 restart files to extract timesteps",
        default="../jorek[0-9]*.h5"
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
    parser.add_argument(
        '-p', '--plot',
        action="store_true",
        help="Whether to plot the data immediately",
        default=False
    )
    args = parser.parse_args()

    
    psi_current_prof_filename = args.exprs_filename
    q_prof_filename = args.qprof_filename
    fourier_file_list = sorted(glob.glob(
        f"exprs_four2d*absolute*n{args.toroidal_mode_number:03.0f}.dat"
    ))
    poloidal_mode_number = args.poloidal_mode_number
    toroidal_mode_number = args.toroidal_mode_number

    h5_filenames = sorted(glob.glob(args.h5_filename_pattern))

    params, jorek_island_data = island_width_from_jorek(
        psi_current_prof_filename,
        q_prof_filename,
        fourier_file_list,
        h5_filenames,
        poloidal_mode_number,
        toroidal_mode_number
    )

    sim_to_disk("jorek_island_data", params, jorek_island_data)

    if args.plot:
        phase_plot(jorek_island_data)
