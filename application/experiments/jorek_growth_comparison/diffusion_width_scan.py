import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from matplotlib import ticker

from jorek_tools.jorek_dat_to_array import (
    read_r_minor
)
from jorek_tools.quasi_linear_model.get_tm_parameters import (
    get_parameters
)
from jorek_tools.quasi_linear_model.central_density_si import (
    central_density_si
)
from jorek_tools.quasi_linear_model.get_diffusion_width import (
    get_diffusion_width
)
from jorek_tools.macroscopic_vars_analysis.plot_quantities import (
    MacroscopicQuantity
)
from tearing_mode_solver.outer_region_solver import (
    solve_system,
    delta_prime_non_linear,
    curvature_stabilisation_non_linear,
    growth_rate_full,
    magnetic_shear,
    alfven_frequency,
    chi_perp_ratio
)

if __name__=='__main__':
    parser = ArgumentParser(
		description="""Solve linear tearing mode outer solution, calculate
        linear growth rate with curvature stabilisation, plot as a function of 
        diffusion width""",
        epilog="Run this script in the `postproc` folder of the simulation " \
            "run to avoid locating exprs_averaged and qprofile files " \
            "manually. Need to run ./jorek2_postproc < get_flux.pp first.",
        formatter_class=ArgumentDefaultsHelpFormatter
	)
    parser.add_argument(
        'resistive_interchange', nargs='+', type=float,
        help="List of resistive interchange values"
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
        '-cm', '--central-mass', type=float,
        help="Central mass (as per JOREK namelist, unitless)",
        default=2.0
    )
    parser.add_argument(
        '-cd', '--central-density', type=float,
        help="Central number density of plasma (10^20/m^3)",
        default=1.0
    )
    parser.add_argument(
        '-si', '--si-units', action='store_true',
        help="Enable this flag to print with SI units. Otherwise, "\
        "results are printed normalised to Alfven frequency",
        default=False
    )
    parser.add_argument(
        '-dat', '--data-file', type=str,
        help="Path to experimental data",
        default=None
    )

    args = parser.parse_args()

    params = get_parameters(
        args.exprs_averaged,
        args.q_profile,
        args.poloidal_mode_number,
        args.toroidal_mode_number
    )

    outer_solution = solve_system(params)


    fig, ax = plt.subplots(1)
    ax.grid()

    shear = magnetic_shear(
        params.q_profile,
        outer_solution.r_s
    )

    gr_conversion = 1.0
    if args.si_units:
        gr_conversion = alfven_frequency(
            params.R0,
            params.B0,
            central_density_si(args.central_mass, args.central_density)
        )

    diff_width = np.linspace(0.015, 0.2, 1000)

    r_minor = read_r_minor(args.exprs_averaged)
    chi_ratios = chi_perp_ratio(
        diff_width,
        outer_solution.r_s,
        params.R0/r_minor,
        params.toroidal_mode_number,
        shear
    )

    # For plotting secondary axis. See 
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/secondary_axis.html
    secax = ax.secondary_xaxis(
        'top',
        functions=(
            lambda x : np.interp(x, diff_width, chi_ratios),
            lambda x : np.interp(x, chi_ratios, diff_width)
        )
    )
    secax.set_xscale('log')
    secax.set_xlabel("$\chi_\perp/\chi_\parallel$")

    for d_r in args.resistive_interchange:
            delta_ps_classical = delta_prime_non_linear(
                outer_solution, 
                0.0
            )
            delta_ps_curv = np.array([curvature_stabilisation_non_linear(
                w_d, 
                d_r, 
                0.0
            ) for w_d in diff_width])
            delta_ps_eff = delta_ps_classical + delta_ps_curv

            delta_ps_eff[delta_ps_eff < 0.0] = 0.0

            growths = [growth_rate_full(
                params.poloidal_mode_number,
                params.toroidal_mode_number,
                params.lundquist_number,
                outer_solution.r_s,
                shear,
                dp
            )*gr_conversion for dp in delta_ps_eff]

            ax.plot(diff_width, growths, label=f'$D_R={d_r:.4f}$')


    if args.data_file:
        mq = MacroscopicQuantity(args.data_file)
        mq.load_x_values_by_index(0)
        mq.load_y_values_by_index(1)
        mq.load_y_errors_by_index(2)

        ax.errorbar(
            mq.x_values,
            mq.y_values,
            yerr=mq.y_errors,
            label="JOREK rMHD",
            capsize=2.0,
            color='black'
        )

    
    # ax2=ax.twiny()
    # ax2.set_xlim(ax.get_xlim())
    # ax2.set_xticks(ax.get_xticks())
    # ax2.set_xbound(ax.get_xbound())

    # def latex_float(f):
    #     float_str = "{0:.1g}".format(f)
    #     if "e" in float_str:
    #         base, exponent = float_str.split("e")
    #         return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    #     else:
    #         return float_str

    # chi_ratio_lambda = lambda x : chi_perp_ratio(
    #     x,
    #     outer_solution.r_s,
    #     params.R0/r_minor,
    #     params.toroidal_mode_number,
    #     shear
    # )
    # ax2.set_xticklabels([
    #     latex_float(chi_ratio_lambda(w)) for w in ax.get_xticks()
    # ])

    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # ax2.xaxis.set_major_formatter(formatter)

    ax.legend()
    ax.set_xlabel("$w_d/a$")
    ax.set_ylabel("Linear growth rate (1/s)")

    plt.show()
