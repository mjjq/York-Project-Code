import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator, UnivariateSpline
import os
import numpy as np
from typing import List, Tuple
import sys
import f90nml

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

def init_flux_fig() -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    fig, ax = plt.subplots(1, figsize=(5, 4))

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Time ($1/\omega_A$)")
    ax.set_ylabel(r"Flux at rational surface ($a^2 B_{\phi 0}$)")

    ax.legend()
    fig.tight_layout()

    return fig, ax

def init_width_fig() -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    fig, ax = plt.subplots(1, figsize=(5, 4))

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Time ($1/\omega_A$)")
    ax.set_ylabel(r"Magnetic island width (a)")

    ax.legend()
    fig.tight_layout()

    return fig, ax

def plot_fluxes(model_times: np.array,
                model_flux: np.array,
                jorek_times: np.array,
                jorek_flux: np.array):
    
    fig, ax = plt.subplots(1, figsize=(5, 4))

    ax.plot(model_times, model_flux, label="Model", color="black")
    ax.plot(jorek_times, jorek_flux, label="JOREK", color="red")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Time ($1/\omega_A$)")
    ax.set_ylabel(r"Flux at rational surface ($a^2 B_{\phi 0}$)")

    ax.legend()
    fig.tight_layout()

    savefig("flux_comparison_log_log")

    ax.set_xscale('linear')
    ax.set_xlim(left=0.0, right=max(jorek_times))

    savefig("flux_comparison_lin_log")

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
                model_dpsi_dt: np.array,
                model_d2psi_dt2: np.array,
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

    # Add factor of 2**(9/4)*r_s to directly relate mode width to island width
    model_mode_widths = 2**(9/4) * r_s * mode_width(
        model_fluxes,
        model_dpsi_dt,
        model_d2psi_dt2,
        r_s,
        params.poloidal_mode_number,
        params.toroidal_mode_number,
        s,
        params.lundquist_number
    )

    jorek_island_widths = island_width(
        jorek_fluxes,
        r_s,
        params.poloidal_mode_number,
        params.toroidal_mode_number,
        s
    )



    fig, ax = plt.subplots(1, figsize=(5, 4))

    ax.plot(
        model_times,
        model_island_widths,
        label="Model ($w(t)$)",
        color="black"
    )
    ax.plot(
        model_times,
        model_mode_widths,
        label=r"Model ($2^{9/4} r_s \delta_{ql}(t)$)",
        color="blue",
        linestyle="dotted"
    )
    ax.plot(
        jorek_times,
        jorek_island_widths,
        label="JOREK ($w(t)$)",
        color="red",
        linestyle="dotted"
    )

    ax.legend()

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Time ($1/\omega_A$)")
    ax.set_ylabel(r"Width ($a$)")

    fig.tight_layout()

    savefig("island_width_comparison_log")

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlim(left=5e5, right=5e6)
    ax.set_ylim(top=1.0)

    savefig("island_width_comparison_linear")


def ql_tm_vs_time():
    """
    Plot various numerically solved variables from a tearing mode solution and
    island width as a function of time from .csv data.
    """
    print(sys.argv)
    if len(sys.argv) < 2:
        jorek_namelist_filename="intear"
        model_data_filename = "./output/05-06-2024_16:42_jorek_model_(m,n)=(2,1).zip"
        jorek_data_filename = "../../jorek_tools/postproc/psi_t_data.csv"
        q_prof_filename = "../../jorek_tools/postproc/qprofile_s00000.dat"
        psi_current_prof_filename = "../../jorek_tools/postproc/exprs_averaged_s00000.csv"
    else:
        jorek_namelist_filename = sys.argv[1]
        q_prof_filename = sys.argv[2]
        psi_current_prof_filename = sys.argv[3]
        jorek_data_filename = sys.argv[4]
        model_data_filename = None
    if len(sys.argv) == 6:
        model_data_filename = sys.argv[5]
    
    poloidal_mode_number = 2.0
    toroidal_mode_number = 1.0

    fig_flux, ax_flux = init_flux_fig()
    fig_width, ax_width = init_width_fig()

    # Load profiles
    q_profile, j_profile = q_and_j_from_csv(psi_current_prof_filename, q_prof_filename)

    # Load jorek data
    jorek_data = pd.read_csv(jorek_data_filename).fillna(0)
    jorek_times, jorek_flux = jorek_flux_at_q(jorek_data, q_profile, 2 / 1)


    jorek_namelist = f90nml.read(jorek_namelist_filename)
    R0 = jorek_namelist['in1']['r_geo']
    F0 = jorek_namelist['in1']['f0']
    B0 = F0/R0

    jorek_times = jorek_to_alfven_time(jorek_times, B0, R0)

    # Load model data if supplied
    min_time = 0.0
    if model_data_filename is not None:    
        params, sol = load_sim_from_disk(model_data_filename)
        poloidal_mode_number = params.poloidal_mode_number
        toroidal_mode_number = params.toroidal_mode_number

        # Minimum time at which our model start relative to the
        # JOREK simulation (such that initial flux of the model 
        # lines up nicely with a flux value in the JOREK run)
        model_initial_flux = sol.psi_t[0]
        jorek_flux_arg = np.argmin((jorek_flux-model_initial_flux)**2)
        jorek_start_time = jorek_times[jorek_flux_arg]

        min_time = np.min(sol.times[sol.times > jorek_start_time])  # 1e4
        max_time = np.max(sol.times)

        # Model flux
        #model_filt = (times < max_time)
        # times = times[model_filt]
        # model_flux = psi_t[model_filt]
        # model_dpsi_dt = dpsi_t[model_filt]
        # model_d2psi_dt2 = d2psi_dt2[model_filt]
        # w_t = w_t[model_filt]

        ax_flux.plot(
            sol.times, sol.psi_t, label='model', color='black'
        )
        ax_width.plot(
            sol.times, sol.w_t, label='model', color='black'
        )

        #ax_flux.set_xlim(left=min_time, right=max_time)
        #ax_width.set_xlim(left=min_time, right=max_time)




    # JOREK flux
    jorek_filt = jorek_times > min_time
    # Subtract min_time to re-zero the sim times
    jorek_times = jorek_times[jorek_filt]-min_time
    jorek_flux = jorek_flux[jorek_filt]

    r_s = rational_surface(q_profile, poloidal_mode_number/toroidal_mode_number)
    s = magnetic_shear(q_profile, r_s)
    jorek_island_widths = island_width(
        jorek_flux,
        r_s,
        poloidal_mode_number,
        toroidal_mode_number,
        s
    )

    ax_flux.plot(jorek_times, jorek_flux, label='JOREK', color='red')
    ax_width.plot(jorek_times, jorek_island_widths, label='JOREK', color='red')

    fig_flux.tight_layout()
    savefig("jorek_flux_comparison")
    fig_width.tight_layout()
    savefig("jorek_width_comparison")

    plt.show()


if __name__ == "__main__":
    ql_tm_vs_time()
    # check_model_t_dependence()
