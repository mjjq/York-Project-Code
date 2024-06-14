import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator, UnivariateSpline
import os
import numpy as np
from typing import List, Tuple
import sys

import imports
from tearing_mode_solver.outer_region_solver import rational_surface, magnetic_shear
from tearing_mode_solver.delta_model_solver import nu, mode_width
from tearing_mode_solver.helpers import (
    classFromArgs,
    TimeDependentSolution,
    savefig,
    load_sim_from_disk,
    TearingModeParameters
)
from tearing_mode_solver.outer_region_solver import island_width
from tearing_mode_solver.algebraic_fitting import get_parab_coefs
from jorek_tools.calc_jorek_growth import growth_rate, _name_time, _name_flux
from jorek_tools.jorek_dat_to_array import q_and_j_from_csv
from jorek_tools.psi_t_from_vtk import jorek_flux_at_q
from jorek_tools.time_conversion import jorek_to_alfven_time


def check_model_t_dependence():
    model_data_filename = "./output/15-05-2024_16:49_jorek_model_(m,n)=(2,1).zip"

    params, sol = load_sim_from_disk(model_data_filename)

    times = sol.times
    psi_t = sol.psi_t
    dpsi_t = sol.dpsi_dt
    w_t = sol.w_t
    d2psi_dt2 = sol.d2psi_dt2
    delta_primes = sol.delta_primes

    print(delta_primes)

    fig, ax = plt.subplots(1)

    ax.plot(times, psi_t)
    ax.set_xscale("log")

    plt.show()

def plot_fluxes(model_times: np.array,
                model_flux: np.array,
                jorek_times: np.array,
                jorek_flux: np.array):
    
    # Parabolic fitting
    # min_t2_time = 1.9e5
    # max_t2_time = 5e5
    # c_0, c_1, c_2 = get_parab_coefs(params, model_flux_func(min_t2_time))

    # t = np.linspace(min_t2_time, max_t2_time, 100)

    fig, ax = plt.subplots(1, figsize=(5, 4))

    ax.plot(model_times, model_flux, label="Model", color="black")
    ax.plot(jorek_times, jorek_flux, label="JOREK", color="red")

    #log_times = np.logspace(4, np.log(max(times)), 100)
    ## ax.plot(log_times, 1.0/log_times, label='2/t dependence')

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Time ($1/\omega_A$)")
    ax.set_ylabel(r"Flux at rational surface ($a^2 B_{\phi 0}$)")

    ## ax.plot(
    ##     t,
    ##     c_0*(t-min_t2_time)**2 +c_1*(t-min_t2_time) + c_2,
    ##     color='green', linestyle='--',
    ##     label=r'$f(t) = at^2 + bt + c$' + \
    ##         f"\n a={c_0:.2e},\n b={c_1:.2e},\n c={c_2:.2e}"
    ## )

    ax.legend()
    fig.tight_layout()

    savefig("flux_comparison_log_log")

    ax.set_xscale('linear')
    ax.set_xlim(left=0.0, right=max(jorek_times))

    savefig("flux_comparison_lin_log")

    #ax.set_xlim(left=40000, right=1e6)
    #ax.set_ylim(bottom=1e-9, top=1e-2)
    #fig.tight_layout()

    #savefig("flux_comparison_log_log_zoom")

    #ax.set_xscale("linear")
    #ax.set_yscale("linear")
    #fig.set_size_inches(6, 4.65, forward=True)
    #ax.set_xlim(left=0, right=4e5)
    #ax.set_ylim(bottom=0, top=0.00175)
    #fig.tight_layout()

    #savefig("flux_comparison_lin_lin_early")

    #ax.set_xlim(left=0, right=1e6)
    #ax.set_ylim(bottom=0, top=0.00175)

    #savefig("flux_comparison_lin_lin")

    #ax.autoscale()
    #ax.set_yscale("log")
    #ax.set_xlim(left=0, right=5e5)
    #savefig("flux_comparison_lin_log")

    #ax.set_xscale("linear")
    #ax.set_yscale("linear")
    #ax.set_xlim(left=0, right=8e4)
    #ax.set_ylim(bottom=0, top=1e-5)
    #savefig("flux_comparison_linear_regime")

def plot_growths(model_times: np.array,
                 model_fluxes: np.array,
                 jorek_times: np.array,
                 jorek_fluxes: np.array):
    model_func = UnivariateSpline(model_times, model_fluxes, s=0)
    model_func_deriv = model_func.derivative()
    model_dpsi_dt = model_func_deriv(model_times)
    model_growths = model_dpsi_dt/model_fluxes

    jorek_func = UnivariateSpline(jorek_times, jorek_fluxes, s=0)
    jorek_func_deriv = jorek_func.derivative()
    jorek_dpsi_dt = jorek_func_deriv(jorek_times)
    jorek_growths = jorek_dpsi_dt/jorek_fluxes

    fig, ax = plt.subplots(1)
    
    ax.plot(model_times, model_growths, label='model', color='black')
    ax.plot(jorek_times, jorek_growths, label='JOREK', color='red')

    ax.set_xscale('log')
    #ax.set_yscale('log')
    
    ax.set_xlabel(r"Time ($1/\omega_A$)")
    ax.set_ylabel(r"Growth rate ($\omega_A$)")

    savefig("growth_rate_comparison")

    return


def plot_widths(model_times: np.array,
                model_fluxes: np.array,
                jorek_times: np.array,
                jorek_fluxes: np.array,
                q_profile: np.array,
                params: TearingModeParameters):
    q_s = params.poloidal_mode_number/params.toroidal_mode_number
    r_s = rational_surface(q_profile, q_s)
    s = magnetic_shear(q_profile, r_s)

    model_island_widths = island_width(
        model_fluxes,
        r_s,
        params.poloidal_mode_number,
        params.toroidal_mode_number,
        s
    )

    jorek_island_widths = island_width(
        jorek_fluxes,
        r_s,
        params.poloidal_mode_number,
        params.toroidal_mode_number,
        s
    )

    fig, ax = plt.subplots(1, figsize=(5, 4))

    ax.plot(model_times, model_island_widths, label="Model", color="black")
    ax.plot(jorek_times, jorek_island_widths, label="JOREK", color="red")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Time ($1/\omega_A$)")
    ax.set_ylabel(r"Magnetic island width ($a$)")

    fig.tight_layout()

    savefig("island_width_comparison")


def ql_tm_vs_time():
    """
    Plot various numerically solved variables from a tearing mode solution and
    island width as a function of time from .csv data.
    """
    print(sys.argv)
    if len(sys.argv) < 3:
        model_data_filename = "./output/05-06-2024_16:42_jorek_model_(m,n)=(2,1).zip"
        jorek_data_filename = "../../jorek_tools/postproc/psi_t_data.csv"
        q_prof_filename = "../../jorek_tools/postproc/qprofile_s00000.dat"
        psi_current_prof_filename = "../../jorek_tools/postproc/exprs_averaged_s00000.csv"
    else:
        model_data_filename = sys.argv[1]
        jorek_data_filename = sys.argv[2]
        q_prof_filename = sys.argv[3]
        psi_current_prof_filename = sys.argv[4]

    q_profile, j_profile = q_and_j_from_csv(psi_current_prof_filename, q_prof_filename)

    params, sol = load_sim_from_disk(model_data_filename)

    times = sol.times
    psi_t = sol.psi_t
    dpsi_t = sol.dpsi_dt
    w_t = sol.w_t
    d2psi_dt2 = sol.d2psi_dt2
    delta_primes = sol.delta_primes

    jorek_data = pd.read_csv(jorek_data_filename).fillna(0)
    jorek_times, jorek_flux = jorek_flux_at_q(jorek_data, q_profile, 2 / 1)
    jorek_times = jorek_to_alfven_time(jorek_times, params.B0, params.R0)

    min_time = np.min(times[times > 0.0])  # 1e4
    max_time = np.max(times)


    print(f"Min time: {min_time}")

    # Model flux
    model_filt = (times < max_time) & (times > min_time)
    times = times[model_filt]
    model_flux = psi_t[model_filt]
    model_flux_func = UnivariateSpline(times, model_flux, s=0)

    # JOREK flux
    jorek_filt = jorek_times > min_time
    jorek_times = jorek_times[jorek_filt]
    jorek_flux = jorek_flux[jorek_filt]

    plot_fluxes(times, model_flux, jorek_times, jorek_flux)
    plot_widths(times, model_flux, jorek_times, jorek_flux, q_profile, params)
    # plot_growths(times, model_flux, jorek_times, jorek_flux)

    plt.show()


if __name__ == "__main__":
    ql_tm_vs_time()
    # check_model_t_dependence()
