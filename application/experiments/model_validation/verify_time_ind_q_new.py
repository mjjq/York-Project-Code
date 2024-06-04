from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from scipy.integrate import quad, simpson
from tqdm import tqdm, trange
from typing import Tuple
import os
import numpy as np
import pandas as pd

import imports
from tearing_mode_solver.profiles import rational_surface
from tearing_mode_solver.y_sol import Y
from tearing_mode_solver.delta_model_solver import nu, mode_width
from tearing_mode_solver.helpers import (
    savefig, classFromArgs, TimeDependentSolution, TearingModeParameters,
    load_sim_from_disk
)
from tearing_mode_solver.outer_region_solver import (
    magnetic_shear, rational_surface,
    island_width, delta_prime_non_linear
)
from tearing_mode_solver.phi_reconstruction import (
    potential, check_solution_is_valid
)

def verify_time_ind_q(sol: TimeDependentSolution,
                params: TearingModeParameters):
    """
    Calculate the full (unapproximated) quasi-linear layer width as a function
    of X and t.

    Parameters:
        sol: TimeDependentSolution
            The time-dependent tearing mode solution
        params: TearingModeParameters
            The tearing mode parameters
        x_range: Tuple[float, float]
            Minimum and maximum bounds defining the range of X values to use in
            the calculation of delta(X,t)
        dx: float
            Distance between adjacent X-values in x_range

    Returns:
        deltas: np.ndarray
            2D array containing delta values as a function of X and t. First
            dimension corresponds to time, second dimension corresponds to
            space.
        times: np.array
            Array of times associated with the first dimension of deltas
        xs: np.array
            Array of X values associated with the second dimension of deltas

    """
    poloidal_mode = params.poloidal_mode_number
    toroidal_mode = params.toroidal_mode_number
    lundquist_number = params.lundquist_number

    r_s = rational_surface(
        params.q_profile,
        poloidal_mode/toroidal_mode
    )
    mag_shear = magnetic_shear(
        params.q_profile,
        r_s
    )


    m=poloidal_mode
    n=toroidal_mode
    S=lundquist_number
    eta = 2e-7 # placeholder for now
    constant_factor = m**2  \
        / (2*n*mag_shear*eta*r_s**2)

    xs = np.linspace(-0.5, 0.5, 100)
    #print(xs)
    
    every = 100
    sol.psi_t = sol.psi_t[::every]
    sol.dpsi_dt = sol.dpsi_dt[::every]
    sol.d2psi_dt2 = sol.d2psi_dt2[::every]
    sol.delta_primes = sol.delta_primes[::every]
    sol.times = sol.times[::every]
    sol.w_t = sol.w_t[::every]
    phi = potential(sol, toroidal_mode, mag_shear, xs)


    dpsif_over_x = np.outer(1.0/xs, sol.psi_t)
    
    condition = constant_factor * phi * dpsif_over_x

    fig, ax = plt.subplots(1, figsize=(5,4))
    times = sol.times
    im = ax.imshow(
        condition,
        extent=[min(times), max(times), min(xs), max(xs)]
    )
    
    fig.colorbar(im)
    ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))

    ax.set_xlabel(r"Normalised time ($1/{\omega}_A$)")
    ax.set_ylabel(r"x ($r_s$)")

    fig.tight_layout()
    
    savefig("time_independent_q")

    

if __name__=='__main__':
    fname = "/home/marcus/Nextcloud/Documents/Fusion CDT/Project/Code/York-Project-Code/application/experiments/jorek_growth_comparison/output/28-05-2024_11:57_jorek_model_(m,n)=(2,1).zip"
    params, sol = load_sim_from_disk(fname)

    verify_time_ind_q(
        sol,
        params
    )

    plt.show()