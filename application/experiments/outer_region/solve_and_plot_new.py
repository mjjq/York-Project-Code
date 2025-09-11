import numpy as np
from matplotlib import pyplot as plt

from tearing_mode_solver.outer_region_solver import delta_prime_non_linear, solve_system, delta_prime
from tearing_mode_solver.helpers import (
    savefig, 
    savecsv, 
    TearingModeParameters,
    sim_to_disk
)

def solution_old_version():
    params = TearingModeParameters(
        poloidal_mode_number = 2,
        toroidal_mode_number = 1,
        lundquist_number = 1e8,
        axis_q = 1.0,
        profile_shaping_factor = 2.0,
        initial_flux = 1e-10
    )

    tm = solve_system(
        params.poloidal_mode_number, 
        params.toroidal_mode_number,
        params.axis_q,
        params.profile_shaping_factor
    )

    return params, tm

def solution_new_version():
    from tearing_mode_solver.profiles import generate_j_profile, generate_q_profile

    q_profile = generate_q_profile(axis_q=1.0, shaping_exponent=2.0)
    j_profile = generate_j_profile(axis_q=1.0, shaping_exponent=2.0)

    params = TearingModeParameters(
        poloidal_mode_number=2,
        toroidal_mode_number=1,
        lundquist_number=1e8,
        initial_flux=1e-12,
        B0=1.0,
        R0=1.0,
        q_profile=q_profile,
        j_profile=j_profile,
        rho0=1.0
    )

    outer_sol = solve_system(params)

    return params, outer_sol

def ql_tm_vs_time():
    """
    Numerically solve the quasi-linear time-dependent tearing mode problem
    and plot the perturbed flux and layer width as functions of time.
    """
    
    params, tm = solution_new_version()

    w_vals = np.linspace(1e-5, 1e-1, 10000)
    dps = delta_prime_non_linear(
        tm, w_vals
    )

    fig, ax0 = plt.subplots(1)
    ax0.plot(w_vals, dps)
    ax0.scatter(tm.r_s-tm.r_range_fwd[-1], delta_prime(tm))
    ax0.grid()
    #savefig("delta_prime")

    # fig1, ax1 = plt.subplots(1)
    # ax1.plot(tm.r_range_fwd, tm.psi_forwards)
    # ax1.plot(tm.r_range_bkwd, tm.psi_backwards)
    # ax1.grid()
    # #savefig("eigenfunction")

    fig2, ax2 = plt.subplots(1)
    ax2.set_xscale('log')
    ax2.grid()
    fwd_r_vals = tm.r_s-tm.r_range_fwd
    ax2.plot(fwd_r_vals, tm.dpsi_dr_forwards, color='black')
    ax2.plot(w_vals/2.0, tm.dpsi_dr_f_func(tm.r_s-w_vals/2.0), linestyle='--', color='red')

    bkwd_r_vals = tm.r_range_bkwd-tm.r_s
    ax2.plot(bkwd_r_vals, tm.dpsi_dr_backwards, color='blue')
    ax2.plot(w_vals/2.0, tm.dpsi_dr_b_func(tm.r_s+w_vals/2.0), linestyle='--', color='orange')

    #ax2.set_xlim(left=9e-5, right=0.02)
    # savefig("dpsi_dr")

    # fig3, ax3 = plt.subplots(1)
    # ax3.plot(tm.r_s-tm.r_range_fwd, tm.r_range_fwd)
    # ax3.plot(tm.r_range_bkwd-tm.r_s, tm.r_range_bkwd)
    # ax3.set_xlim(left=9e-5, right=0.02)
    # ax3.set_ylim(bottom=0.795, top=0.797)
    # ax3.set_xscale('log')
    # ax3.grid()
    # savefig("eigenfunction_dist")

    plt.show()
    return

if __name__=='__main__':
    ql_tm_vs_time()