from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

from tearing_mode_solver.outer_region_solver import solve_system, TearingModeParameters, compute_derivatives

from tearing_mode_solver.profiles import poly_profiles_coef, poly_profiles_loc


def test_outer_solution_satifies_ode():
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

    r_vals, j_vals = zip(*j_profile)
    j_vals = j_vals/j_vals[0]

    j_func = UnivariateSpline(r_vals, j_vals, s=0.0)
    dj_dr_func = j_func.derivative()

    r_vals, q_vals = zip(*q_profile)

    q_func = UnivariateSpline(r_vals, q_vals, s=0.0)

    sol = solve_system(params)


    m = params.poloidal_mode_number
    n = params.toroidal_mode_number

    dj_dr = dj_dr_func(sol.r_range_fwd)
    q = q_func(sol.r_range_fwd)
    q0 = q_func(0.0)
    A = 2.0*(q/q0*m*dj_dr)/(n*q - m)
    
    d2psi_dr2_func = UnivariateSpline(sol.r_range_fwd, sol.dpsi_dr_forwards, s=0).derivative()
    d2psi_dr2 = d2psi_dr2_func(sol.r_range_fwd)

    rhs = sol.psi_forwards*(m**2 - A*sol.r_range_fwd) + sol.r_range_fwd*sol.dpsi_dr_forwards
    lhs = sol.r_range_fwd**2 * d2psi_dr2

    err = rhs-lhs
    
    fig, ax = plt.subplots(3)
    ax_lhs, ax_rhs, ax_err = ax
    ax_lhs.plot(sol.r_s-sol.r_range_fwd,lhs)
    ax_rhs.plot(sol.r_s-sol.r_range_fwd,rhs)
    ax_err.plot(sol.r_s-sol.r_range_fwd,err)
    plt.show()


if __name__=='__main__':
    test_outer_solution_satifies_ode()
