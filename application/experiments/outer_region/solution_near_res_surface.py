from matplotlib import pyplot as plt

import imports
from tearing_mode_solver.outer_region_solver import *

def solution_near_res_surface():
    """
    Plot the perturbed flux of the outer solution as a function of plasma minor
    radius near the resonant surface of the plasma.
    """
    poloidal_mode = 3
    toroidal_mode = 2
    axis_q = 1.0
    resolution = 1e-5
    layer_width = 1e-4

    tm = solve_system(
        poloidal_mode, toroidal_mode, axis_q, resolution, layer_width
    )

    delta_p = delta_prime(tm)

    print(f"Delta prime = {delta_p}")

    fig, ax = plt.subplots(1, figsize=(4,3), sharex=True)

    #ax4.plot(r_range_fwd, dpsi_dr_forwards)
    #ax4.plot(r_range_bkwd, dpsi_dr_backwards)

    ax.scatter(
        tm.r_range_fwd, tm.psi_forwards, label='Solution below $\hat{r}_s$'
    )
    ax.scatter(
        tm.r_range_bkwd, tm.psi_backwards, label='Solution above $\hat{r}_s$'
    )
    rs_line = ax.vlines(
         tm.r_s, ymin=0.0,
         ymax=np.max(np.concatenate((tm.psi_forwards, tm.psi_backwards))),
         linestyle='--', color='red',
         label=f'Rational surface $\hat{{q}}(\hat{{r}}_s) = {poloidal_mode}/{toroidal_mode}$'
    )

    ax.set_xlabel("Normalised minor radial co-ordinate (r/a)")
    ax.set_ylabel(r"Normalised perturbed flux $\hat{\delta \psi}^{(1)}$")

    ax.set_xlim(
        left = tm.r_s - 2.0*layer_width,
        right = tm.r_s + 2.0*layer_width
    )

    fig.tight_layout()

    plt.show()

if __name__=='__main__':
    solution_near_res_surface()