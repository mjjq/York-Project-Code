from dataclasses import dataclass
import numpy as np

from jorek_tools.delta_psi_extraction.plot_delta_psi_vs_time import get_psi_vs_time_for_mode
from jorek_tools.jorek_dat_to_array import read_four2d_profile, read_q_profile, read_timestep_map
from tearing_mode_solver.helpers import TimeDependentSolution

from experiments.ntm_modelling.mre_time_series import MeasuredIslandWidth

@dataclass
class IslandCalibrations:
    # List of times at which calibrations took place
    times: np.array
    # Inner radial extent of the island X-point surface
    # in s=sqrt(psi_N) co-ords
    rho_min_avg: np.array
    # Outer radial extent of the island X-point surface
    # in s=sqrt(psi_N) co-ords
    rho_max_avg: np.array
    # Average island width (rho_max-rho_min), 
    # normalised to minor radius
    w_avg: np.array

def read_island_width_calibrations(filename: str) -> IslandCalibrations:
    data = np.loadtxt(filename)
    times, rhomin, rhomax = data[:,0], data[:,1], data[:,2]

    w_avg = rhomax-rhomin

    return IslandCalibrations(
        times,
        rhomin,
        rhomax,
        w_avg
    )

def get_calibrated_island_width_series(delta_psi_sol: TimeDependentSolution,
                                       calibrations: IslandCalibrations,
                                       debug_plot: bool = False) -> TimeDependentSolution:
    """
    Use calibrations to fully determine island width as a function
    of time with delta_psi measurements
    """
    dpsi_calibs = np.interp(
        calibrations.times, delta_psi_sol.times, delta_psi_sol.psi_t
    )

    log_dpsi, log_widths = np.log(dpsi_calibs), np.log(calibrations.w_avg)

    coefs = np.polyfit(log_dpsi, log_widths, 1)
    print(coefs)
    func = np.poly1d(coefs)
    log_psi_t = np.log(delta_psi_sol.psi_t)
    log_calibrated_w_avg = func(log_psi_t)
    calibrated_w_avg = np.exp(log_calibrated_w_avg)

    if debug_plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, figsize=(5,4))
        ax.set_xlabel("$\delta\psi(r_s)$ (arb)")
        ax.set_ylabel("$w_{avg}/a$")
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.plot(dpsi_calibs, calibrations.w_avg, label="Calibrations")


        ax.plot(delta_psi_sol.psi_t, calibrated_w_avg, label=f"Fit (A,B={coefs[0]:.2g}, {coefs[1]:.2g})")
        ax.legend()
        ax.grid()
        fig.tight_layout()
        plt.show()

    return TimeDependentSolution(
        delta_psi_sol.times,
        delta_psi_sol.psi_t,
        delta_psi_sol.dpsi_dt,
        delta_psi_sol.d2psi_dt2,
        calibrated_w_avg,
        delta_psi_sol.delta_primes
    )

    

def plot_calibration_main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Plot delta psi at a chosen rational "
        "surface as a function of time"
    )
    parser.add_argument(
        '-f', '--fourier-data',
        help='List of fourier data postproc files',
        nargs='+'
    )
    parser.add_argument(
        '-q', '--qprofile-filename',
        help="Location of the q-profile file extracted using jorek2_postproc",
        type=str
    )
    parser.add_argument(
        '-t', '--time-map-filename',
        help='Location of file containing map between timestep and SI time',
        type=str
    )
    parser.add_argument(
        '-w', '--island-calibrations',
        help="Text file containing island width calibrations at various times"
    )
    parser.add_argument(
        '-m', '--poloidal-mode',
        help="List of poloidal mode numbers to evaluate",
        default=2,
        type=int
    )
    parser.add_argument(
        '-n', '--toroidal-mode', type=int,
        help='Toroidal mode number',
        default=1,
    )
    parser.add_argument(
        '-p', '--plot-calibration', action='store_true',
        help="Use this flag to enable debug plotting"
    )
    parser.add_argument(
        '-d', '--debug-plot', action='store_true',
        help="Enable debug plotting (plot delta_psi vs calibrated w)"
    )

    args = parser.parse_args()

    fourier_data = [read_four2d_profile(fname) for fname in args.fourier_data]
    qprofile = read_q_profile(args.qprofile_filename)
    tstep_map = read_timestep_map(args.time_map_filename)

    sol = get_psi_vs_time_for_mode(
        fourier_data,
        [args.poloidal_mode],
        args.toroidal_mode,
        qprofile,
        tstep_map
    )[0]

    calibrations = read_island_width_calibrations(args.island_calibrations)

    sol_calib = get_calibrated_island_width_series(sol, calibrations, args.debug_plot)

    if args.plot_calibration:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, figsize=(5,4))
        ax.plot(sol_calib.times, sol_calib.w_t)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$w_{avg}/a$")
        ax.grid()
        fig.tight_layout()
        plt.show()

    measured_width = MeasuredIslandWidth(
        sol_calib.times,
        sol_calib.w_t,
        np.zeros(len(sol_calib.times)),
        normalised=True
    )
    measured_width.write("w_measured.txt")


if __name__=='__main__':
    plot_calibration_main()
