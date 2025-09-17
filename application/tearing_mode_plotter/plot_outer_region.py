from matplotlib import pyplot as plt
import numpy as np

from tearing_mode_solver.outer_region_solver import (
    OuterRegionSolution,
    delta_prime
)
from tearing_mode_solver.helpers import savefig

def plot_outer_region_solution(tm: OuterRegionSolution):

    dp = delta_prime(tm)
    print(f"r_s Delta' = {tm.r_s * dp}")

    fig, axs = plt.subplots(3)
    ax, ax_dpsi_dr, ax_dpsi_dr_dist = axs
    
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

    ax_dpsi_dr_dist.plot(

    )

    
    ax.set_ylabel("Normalised perturbed flux")
    ax_dpsi_dr.set_ylabel("Normalised $\partial \delta\psi^{(1)}/\partial r$") 
    ax_dpsi_dr.set_xlabel("Normalised minor radial co-ordinate (a)")
    
    fig.legend()
    fig.tight_layout()

    savefig("outer_region")