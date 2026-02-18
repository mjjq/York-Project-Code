import argparse
from typing import List, Tuple, Optional
import numpy as np
from matplotlib import pyplot as plt

from jorek_tools.jorek_dat_to_array import (
    read_four2d_profile, filter_four2d_mode,
    Four2DProfile, read_q_profile, read_timestep_map,
    TimestepMap
)
from tearing_mode_solver.outer_region_solver import rational_surface
from tearing_mode_solver.helpers import TimeDependentSolution

def get_psi_at_psi_s(prof: Four2DProfile,
                     psi_s: float) -> float:
    return np.interp(psi_s, prof.psi_norm, prof.psi)

def get_psi_vs_time_for_mode(four2d_profs: List[List[Four2DProfile]],
                             poloidal_mode_numbers: List[int],
                             toroidal_mode_number: int,
                             qprofile: Optional[List[Tuple[float, float]]] = None,
                             tstep_map: Optional[TimestepMap] = None) -> List[TimeDependentSolution]:
    """
    Get delta_psi(r_s) as a function of time from a set of fourier
    decomposed JOREK .dat files.

    Each element in the return list corresponds to delta_psi(t) for
    the corresponding element in poloidal_mode_numbers.

    If qprofile_filename is not specified, delta_psi is evaluated
    at psi_N=0.5.

    If timestep_map_filename is not specified, the time axis is mapped
    to JOREK time step instead.

    """
    rational_surfaces = [0.5 for m in poloidal_mode_numbers]
    if qprofile:
        rational_surfaces = [
            rational_surface(qprofile, m/toroidal_mode_number)
            for m in poloidal_mode_numbers
        ]

    print(f"Rational surfaces (psi_N): {['{:.3g}'.format(i) for i in rational_surfaces]}")

    dpsi_vs_time_data = []
    times = []

    for prof_timeslice in four2d_profs:
        modes: List[Four2DProfile] = prof_timeslice

        psi_at_rs_timeslice = []
        for i,poloidal_mode_number in enumerate(poloidal_mode_numbers):
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

    sols: List[TimeDependentSolution] = []

    dpsi_vs_time_data = np.array(dpsi_vs_time_data)
    for i,mode in enumerate(poloidal_mode_numbers):
        psi_t = dpsi_vs_time_data[:,i]
        dpsi_dt = np.diff(psi_t)/np.diff(times)
        d2psi_dt2 = np.diff(dpsi_dt)/np.diff(times[:-1])
        
        sols.append(TimeDependentSolution(
            times=times,
            psi_t=psi_t,
            dpsi_dt=dpsi_dt,
            d2psi_dt2=d2psi_dt2,
            w_t=None,
            delta_primes=None
        ))

    return sols


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

    fourier_data = [read_four2d_profile(fname) for fname in args.fourier_data]
    qprofile = None
    if args.qprofile_filename:
        qprofile = read_q_profile(args.qprofile_filename)
    tstep_map = None
    if args.time_map_filename:
        tstep_map = read_timestep_map(args.time_map_filename)



    sols = get_psi_vs_time_for_mode(
        fourier_data,
        args.poloidal_modes,
        args.toroidal_mode,
        qprofile,
        tstep_map
    )

    fig, ax = plt.subplots(1, figsize=(5,4))

    for i,mode in enumerate(args.poloidal_modes):
        psi_vs_time = sols[i].psi_t
        times = sols[i].times
        ax.plot(times, psi_vs_time, label=f'm={mode}')

    ax.legend()
    ax.grid()
    ax.set_yscale('log')

    ax.set_xlabel('Time step (arb)')
    if args.time_map_filename:
        ax.set_xlabel("Time (s)")

    ax.set_ylabel(r"$\delta\psi(r_s)$ (arb)")

    fig.tight_layout()
    plt.show()
