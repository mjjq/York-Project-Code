from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt

from tearing_mode_solver.outer_region_solver import solve_system, TearingModeParameters, delta_prime

from tearing_mode_solver.profiles import poly_profiles_coef, poly_profiles_loc




if __name__=='__main__':
    q_0 = 1.0
    q_shape = 1.0
    q_profile, j_profile = poly_profiles_coef([1.5, 0.0, 1.0])
    q_profile, j_profile = poly_profiles_loc(1.8, 2.87, 2.0, 0.5)


    params = TearingModeParameters(
        poloidal_mode_number=2,
        toroidal_mode_number=1,
        lundquist_number=1e7,
        initial_flux=1e-12,
        B0=1.0,
        R0=10.0,
        q_profile=q_profile,
        j_profile=j_profile
    )

    sol = solve_system(params)

    print("r_s Delta': ", sol.r_s*delta_prime(sol))

    plt.scatter(sol.r_range_fwd, sol.psi_forwards)
    plt.scatter(sol.r_range_bkwd, sol.psi_backwards)

    print(sol.r_range_fwd[-1])

    plt.show()
