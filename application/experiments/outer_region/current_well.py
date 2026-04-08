import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from scipy.special import erf
from typing import Tuple

from tearing_mode_solver.outer_region_solver import delta_prime_non_linear, solve_system, delta_prime
from tearing_mode_solver.helpers import (
    savefig, 
    savecsv, 
    TearingModeParameters,
    sim_to_disk
)
from tearing_mode_plotter.plot_outer_region import plot_outer_region_solution

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

def j_profile_gauss_mod(gauss_amp: float,
                        gauss_loc: float,
                        gauss_width: float,
                        equil_width: float) -> Tuple[np.array, np.array]:
    x = np.linspace(0.0, 1.0, 10000)

    j = np.exp(-(x/equil_width)**2) + gauss_amp * np.exp(
        -(x-gauss_loc)**2 / gauss_width**2
    )

    return list(zip(x, j))

def q_profile_gauss_mod(gauss_amp: float,
                        gauss_loc: float,
                        gauss_width: float,
                        equil_width: float,
                        q0: float) -> Tuple[np.array, np.array]:
    x = np.linspace(0.0, 1.0, 10000)

    A = gauss_amp
    a = equil_width
    b = gauss_width
    x0 = gauss_loc
    # B_theta = 
    integral_jphi = -0.5*a**2 * np.exp(-(x/a)**2) + 0.5*A*b*(
        np.sqrt(np.pi) * x0 * erf((x-x0)/b) - b * np.exp(-(x-x0)**2/b**2)
    )

    const = 0.5*a**2 - 0.5*A*b*(
        x0*np.sqrt(np.pi)*erf(-x0/b) -
        b*np.exp(-x0**2/b**2)
    )

    integral_jphi = integral_jphi + const
    
    q = x**2 / integral_jphi

    q = q0/q[1] * q
    q[0]=q0

    return list(zip(x, q))

def solution_new_version():
    from tearing_mode_solver.profiles import generate_j_profile, generate_q_profile

    A, x0, b, a = 0.5, 0.7, 0.125, 0.53
    j_profile = j_profile_gauss_mod(A, x0, b, a)
    q_profile = q_profile_gauss_mod(A, x0, b, a, 1.45)

    # fig, axs = plt.subplots(2)
    # r, q = zip(*q_profile)
    # r, j = list(zip(*j_profile))
    # axs[0].plot(r, j)
    # axs[1].plot(r, q)
    # plt.show()

    params = TearingModeParameters(
        poloidal_mode_number=2,
        toroidal_mode_number=1,
        lundquist_number=1e8,
        initial_flux=1e-13,
        B0=1.0,
        R0=1.0,
        q_profile=q_profile,
        j_profile=j_profile,
        rho0=1.0
    )

    outer_sol = solve_system(params, resolution=1e-5, r_s_thickness=1e-6)

    print("Finished outer solution")

    return params, outer_sol

def ql_tm_vs_time():
    """
    Numerically solve the quasi-linear time-dependent tearing mode problem
    and plot the perturbed flux and layer width as functions of time.
    """
    parser = ArgumentParser(
        description="Plot outer region solution with debugging"
    )
    parser.add_argument(
        '-ex', '--exprs-filename', type=str,
        help="Use JOREK postproc expressions as input",
        default=None
    )
    parser.add_argument(
        '-q', '--qprofile-filename', type=str,
        help="Use JOREK qprofile as input",
        default=None
    )
    parser.add_argument(
        '-c', '--chease-filename', type=str,
        help='Use chease columns as input',
        default=None
    )
    parser.add_argument(
        '-qm', '--q-scale-factor', type=float,
        help="Constant scale factor for the safety factor profile",
        default=1.0
    )
    args=parser.parse_args()

    if args.exprs_filename:
        from jorek_tools.quasi_linear_model.get_tm_parameters import get_parameters
        params = get_parameters(
            args.exprs_filename,
            args.qprofile_filename,
            2, 1
        )
        rs, qs = zip(*params.q_profile)
        qs = args.q_scale_factor * np.array(qs)
        params.q_profile = list(zip(rs, qs))
        tm = solve_system(params)
    elif args.chease_filename:
        from chease_tools.dr_term_at_q import read_columns
        from chease_tools.get_tm_parameters import get_parameters
        cols = read_columns(args.chease_filename)
        params = get_parameters(
            cols,
            2,1
        )
        tm = solve_system(params)
    else:
        params, tm = solution_new_version()

    w_vals = [1e-7]
    dps = delta_prime(
        tm, w_vals
    )

    fig, ax0 = plt.subplots(1)
    ax0.grid()
    ax0.set_xlabel("w/a")
    ax0.set_ylabel("$a\Delta'(w)$")
    ax0.plot(w_vals, dps, color='black')
    #ax0.scatter(tm.r_s-tm.r_range_fwd[-1], delta_prime(tm))
    #savefig("delta_prime")

    # fig1, ax1 = plt.subplots(1)
    # ax1.plot(tm.r_range_fwd, tm.psi_forwards)
    # ax1.plot(tm.r_range_bkwd, tm.psi_backwards)
    # ax1.grid()
    # #savefig("eigenfunction")


    #fig2, ax2 = plt.subplots(1)
    #ax2.set_xscale('log')
    #ax2.grid()
    #fwd_r_vals = tm.r_s-tm.r_range_fwd
    #ax2.plot(fwd_r_vals, tm.dpsi_dr_forwards, color='black', marker='x')
    #ax2.scatter(w_vals/2.0, tm.dpsi_dr_f_func(tm.r_s-w_vals/2.0), color='red')

    #bkwd_r_vals = tm.r_range_bkwd-tm.r_s
    #offset = tm.dpsi_dr_forwards[-1]-tm.dpsi_dr_backwards[-1]
    #ax2.plot(bkwd_r_vals, offset + tm.dpsi_dr_backwards, color='blue', marker='x')
    
    #ax2.scatter(w_vals/2.0, tm.dpsi_dr_b_func(tm.r_s+w_vals/2.0), color='orange')

    plot_outer_region_solution(tm)

    fig3, ax3 = plt.subplots(2, sharex=True)
    axj, axq = ax3
    axj.grid(); axq.grid()

    axj.set_ylabel("$J_\phi$ (A/m$^2$)")
    axq.set_ylabel("q")

    axq.set_xlabel("r/a")

    rs, js = zip(*params.j_profile)
    axj.plot(rs, js, color='black')
    axj.vlines(tm.r_s, min(js), max(js), color='red', linestyle='--')

    rs, qs = zip(*params.q_profile)
    axq.plot(rs, qs, color='black')
    axq.vlines(tm.r_s, min(qs), max(qs), color='red', linestyle='--')

    plt.show()
    return

if __name__=='__main__':
    ql_tm_vs_time()
