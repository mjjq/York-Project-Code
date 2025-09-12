import argparse
from typing import List
import numpy as np
from matplotlib import pyplot as plt

from jorek_tools.jorek_dat_to_array import (
    read_four2d_profile, filter_four2d_mode,
    Four2DProfile
)

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
        help='Fourier data postproc file',
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

    modes: List[Four2DProfile] = read_four2d_profile(args.fourier_data)

    fig, ax = plt.subplots(1, figsize=(4,3))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    psi_at_rs_timeslice = []
    for i,poloidal_mode_number in enumerate(args.poloidal_modes):
        prof = filter_four2d_mode(modes, poloidal_mode_number)
        ax.plot(prof.psi_norm, prof.psi, label=f'm={poloidal_mode_number}')

    ax.legend()
    ax.grid()
    ax.set_xlabel(r'$\psi_N$')
    ax.set_ylabel(r"$\delta\psi$ (arb)")

    fig.tight_layout()
    plt.show()