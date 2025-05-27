from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from jorek_tools.jorek_dat_to_array import (
    read_r_minor,
    read_chi_par_profile_rminor,
    read_chi_perp_profile_rminor
)
from jorek_tools.quasi_linear_model.get_tm_parameters import get_parameters
from tearing_mode_solver.profiles import value_at_r
from tearing_mode_solver.outer_region_solver import (
    rational_surface,
    diffusion_width,
    magnetic_shear,
)

def get_diffusion_width(exprs_averaged_filename: str,
                        q_profile_filename: str,
                        poloidal_mode_number: int,
                        toroidal_mode_number: int) -> float:
    """
    Get diffusion width associated with a particular rational surface
    from JOREK postproc data.

    :param exprs_averaged_filename: Postproc filename. See get_flux.pp for format.
    :param q_profile_filename: Name of q-profile file extracted from postproc.
        See get_flux.pp for format.
    :param poloidal_mode_number: Poloidal mode associated with the rational surface.
    :param toroidal_mode_number: Toroidal mode associated with the rational surface.

    :return: Diffusion width w_d, normalised to plasma minor radius.
    """
    params = get_parameters(
        exprs_averaged_filename,
        q_profile_filename,
        poloidal_mode_number,
        toroidal_mode_number
    )

    r_minor = read_r_minor(exprs_averaged_filename)
    r_s = rational_surface(
        params.q_profile, 
        params.poloidal_mode_number/params.toroidal_mode_number
    )
    r_s_si = r_minor*r_s

    chi_perp_profile = read_chi_perp_profile_rminor(exprs_averaged_filename)
    chi_perp_rs = value_at_r(chi_perp_profile, r_s_si)

    chi_par_profile = read_chi_par_profile_rminor(exprs_averaged_filename)
    chi_par_rs = value_at_r(chi_par_profile, r_s_si)

    mag_shear = magnetic_shear(params.q_profile, r_s)

    return diffusion_width(
        chi_perp_rs,
        chi_par_rs,
        r_s,
        params.R0/r_minor,
        params.toroidal_mode_number,
        mag_shear
    )

if __name__=='__main__':
    parser = ArgumentParser(
		description="""Get diffusion width from JOREK inputs""",
        epilog="Run this script in the `postproc` folder of the simulation " \
            "run to avoid locating exprs_averaged and qprofile files " \
            "manually. Need to run ./jorek2_postproc < get_flux.pp first.",
        formatter_class=ArgumentDefaultsHelpFormatter
	)
    parser.add_argument(
        '-ex', '--exprs-averaged',  type=str, default="exprs_averaged_s00000.dat",
        help="Path to exprs_averaged...dat postproc file (Optional)"
    )
    parser.add_argument(
        '-q', '--q-profile', type=str, default="qprofile_s00000.dat",
        help="Path to qprofile...dat file (Optional)"
    )
    parser.add_argument(
        '-m', '--poloidal-mode-number', type=int, default=2,
        help="Poloidal mode number of the tearing mode"
    )
    parser.add_argument(
        '-n', '--toroidal-mode-number', type=int, default=1,
        help="Toroidal mode number of the tearing mode"
    )
    parser.add_argument(
        '-si', '--si-units', action='store_true',
        help="Enable this flag to print with SI units. Otherwise, "\
        "results are printed normalised to minor radius",
        default=False
    )

    args = parser.parse_args()

    w_d = get_diffusion_width(
        args.exprs_averaged,
        args.q_profile,
        args.poloidal_mode_number,
        args.toroidal_mode_number
    )
    
    if args.si_units:
        r_minor = read_r_minor(args.exprs_averaged)
        w_d = w_d * r_minor

    print(w_d)