from argparse import ArgumentParser
from matplotlib import pyplot as plt

from tearing_mode_solver.helpers import load_sim_from_disk

from tearing_mode_plotter.plot_magnetic_island_width import island_width_plot, phase_plot
from tearing_mode_plotter.plot_quasi_linear_solution import plot_perturbed_flux, plot_growth_rate

def get_quantities():
    return [
        'flux',
        'growth',
        'w',
        'phase'
    ]

if __name__=='__main__':
    parser = ArgumentParser(
        description="Plot quantities from a quasi-linear solver data file (.zip format)"
    )

    parser.add_argument(
        "filename", 
        help="Path to the solution file (.zip format)",
        type=str
    )

    parser.add_argument(
        "-p", "--plot",
        help="Quantities to plot. (Use -l to list quantities)",
        default="all",
        nargs="+"
    )

    parser.add_argument(
        "-l", '--list-quantities',
        help="List plottable quantities",
        action="store_true"
    )

    parser.add_argument(
        "-si", "--si-time",
        help="Plot time-related quantities with SI-units",
        action='store_true'
    )

    args = parser.parse_args()

    if args.list_quantities:
        print()
        print("Available quantities:")
        print("\n".join(get_quantities()))
        print()

        exit()

    params, sol = load_sim_from_disk(args.filename)


    if 'all' in args.plot:
        plot_perturbed_flux(params, sol, args.si_time)
        plot_growth_rate(params, sol, args.si_time)
        island_width_plot(params, sol, args.si_time)
        phase_plot(params, sol, args.si_time)

        plt.show()

        exit(0)

    if 'flux' in args.plot:
        plot_perturbed_flux(params, sol, args.si_time)
    if 'growth' in args.plot:
        plot_growth_rate(params, sol, args.si_time)
    if 'w' in args.plot:
        island_width_plot(params, sol, args.si_time)
    if 'phase' in args.plot:
        phase_plot(params, sol, args.si_time)

    plt.show()