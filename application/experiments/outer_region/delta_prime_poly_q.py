from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt

from tearing_mode_solver.outer_region_solver import solve_system, TearingModeParameters, delta_prime

def poly(coefficients: np.array, n_points: int = 50000) -> np.array:
    r = np.linspace(0.0, 1.0, n_points)
    return r, coefficients[0] + np.sum([coef*r**k for k,coef in enumerate(coefficients[1:],start=1)], axis=0)

def poly_prime(coefficients: np.array, n_points: int = 50000) -> np.array:
    r = np.linspace(0.0, 1.0, n_points)
    return r, coefficients[1] + np.sum([k*coef*r**(k-1) for k,coef in enumerate(coefficients[2:],start=2)], axis=0)

def poly_profiles_coef(coefs: np.array):
    r, q = poly(coefs)
    r, q_prime = poly_prime(coefs)

    j = 2.0*(1.0-r*q_prime/q)/q

    q_profile = list(zip(r,q))
    j_profile = list(zip(r,j))

    return q_profile, j_profile

def poly_profiles_loc(q_0: float, q_edge: float, q_s: float, r_s: float):
    c = (q_s - q_0 - (q_edge-q_0)*r_s**2)/(r_s**4 - r_s**2)
    b = q_edge - q_0 - c

    coefs = [q_0, 0.0, b, 0.0, c]
    return poly_profiles_coef(coefs)




if __name__=='__main__':
    q_0 = 1.0
    q_shape = 1.0
    q_profile, j_profile = poly_profiles_coef([1.5, 0.0, 1.0])
    q_profile, j_profile = poly_profiles_loc(1.8, 2.9, 2.0, 0.5)


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
