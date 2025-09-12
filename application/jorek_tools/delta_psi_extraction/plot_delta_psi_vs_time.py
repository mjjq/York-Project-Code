import argparse
from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt

from jorek_tools.jorek_dat_to_array import (
    read_four2d_profile, filter_four2d_mode,
    Four2DProfile, read_q_profile, read_timestep_map
)
from tearing_mode_solver.outer_region_solver import rational_surface

def get_psi_at_psi_s(prof: Four2DProfile,
                     psi_s: float) -> float:
    return np.interp(psi_s, prof.psi_norm, prof.psi)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
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
        '-m', '--poloidal-modes',
        help="List of poloidal mode numbers to evaluate",
        nargs='+',
        default=[1,2,3,4,5],
        type=int
    )
    parser.add_argument(
        '-n', '--toroidal-mode', type=int,
        help='Toroidal mode number',
        default=1,
    )

    args = parser.parse_args()


    rational_surfaces = [0.5 for m in args.poloidal_modes]
    if args.qprofile_filename:
        q_prof = read_q_profile(args.qprofile_filename)
        rational_surfaces = [
            rational_surface(q_prof, m/args.toroidal_mode)
            for m in args.poloidal_modes
        ]

    print(f"Rational surfaces (psi_N): {['{:.3g}'.format(i) for i in rational_surfaces]}")

    tstep_map = None
    if args.time_map_filename:
        tstep_map = read_timestep_map(args.time_map_filename)

    dpsi_vs_time_data = []
    times = []

    for prof_timeslice in args.fourier_data:
        modes: List[Four2DProfile] = read_four2d_profile(prof_timeslice)

        psi_at_rs_timeslice = []
        for i,poloidal_mode_number in enumerate(args.poloidal_modes):
            prof = filter_four2d_mode(modes, poloidal_mode_number)
            psi_at_rs = get_psi_at_psi_s(prof, rational_surfaces[i])

            psi_at_rs_timeslice.append(psi_at_rs)

        dpsi_vs_time_data.append(psi_at_rs_timeslice)
        
        if tstep_map:
            times.append(
                np.interp(
                    modes[0].timestep, 
                    tstep_map.time_steps, 
                    tstep_map.times
                )
            )
        else:
            times.append(modes[0].timestep)



    fig, ax = plt.subplots(1, figsize=(4,3))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    dpsi_vs_time_data = np.array(dpsi_vs_time_data)
    for i,mode in enumerate(args.poloidal_modes):
        psi_vs_time = dpsi_vs_time_data[:,i]
        ax.plot(times, psi_vs_time, label=f'm={mode}')

    ax.legend()
    ax.grid()

    ax.set_xlabel('Time step (arb)')
    if tstep_map:
        ax.set_xlabel("Time (s)")

    ax.set_ylabel(r"$\delta\psi$ (arb)")

    fig.tight_layout()
    plt.show()