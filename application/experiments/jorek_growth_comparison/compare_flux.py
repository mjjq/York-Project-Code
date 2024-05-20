import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator, UnivariateSpline
import os
import numpy as np
from typing import List, Tuple

import imports
from tearing_mode_solver.outer_region_solver import (
    rational_surface, magnetic_shear
)
from tearing_mode_solver.delta_model_solver import nu, mode_width
from tearing_mode_solver.helpers import (
    classFromArgs, TimeDependentSolution, savefig, load_sim_from_disk
)
from tearing_mode_solver.outer_region_solver import island_width
from tearing_mode_solver.algebraic_fitting import get_parab_coefs
from jorek_tools.calc_jorek_growth import growth_rate, _name_time, _name_flux
from jorek_tools.jorek_dat_to_array import q_and_j_from_csv


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
    ax.set_xscale('log')

    plt.show()
    
def jorek_flux_interp_func(jorek_psi_t_data: pd.DataFrame) \
    -> CloughTocher2DInterpolator:
    """
    Get temporal evolution of flux at a particular radial co-ordinate.

    Parameters
    ----------
    jorek_psi_t_data : pd.DataFrame
        Dataframe containing perturbed flux as a function of r and t.
        
    Returns
    -------
    CloughTocher2DInterpolator
        Psi(r, t)

    """
    grouped = jorek_psi_t_data.groupby('time')
    
    vals = []
    coords = []
    
    for time, group in grouped:
        vals += list(group['Psi'])
        coords += [(time, r_val) for r_val in group['r']]
        
    vals = np.array(vals)
    coords = np.array(coords)
    return CloughTocher2DInterpolator(coords, vals)
        
def r_from_q(q_profile: List[Tuple[float, float]],
             target_q: float):
    rs, qs = zip(*q_profile)
    
    spline = UnivariateSpline(qs, rs, s=0)
    
    return spline(target_q)
    
    

def jorek_flux_at_q(jorek_data: pd.DataFrame,
                    q_profile: List[Tuple[float, float]],
                    target_q: float) -> Tuple[List[float], List[float]]:
    target_r = r_from_q(q_profile, target_q)
    
    jorek_psi_t_func = jorek_flux_interp_func(jorek_data)
    
    times = np.unique(jorek_data['time'])
    
    return np.array(times), \
        np.array([jorek_psi_t_func(t, target_r) for t in times])
    

def ql_tm_vs_time():
    """
    Plot various numerically solved variables from a tearing mode solution and
    island width as a function of time from .csv data.
    """
    model_data_filename = "./output/17-05-2024_19:26_jorek_model_(m,n)=(2,1).zip"
    jorek_data_filename = "../../jorek_tools/postproc/psi_t_data.csv"
    q_prof_filename = "../../jorek_tools/postproc/qprofile_s00000.dat"
    psi_current_prof_filename = "../../jorek_tools/postproc/exprs_averaged_s00000.csv"
    q_profile, j_profile = q_and_j_from_csv(
        psi_current_prof_filename, q_prof_filename
    )


    params, sol = load_sim_from_disk(model_data_filename)

    times = sol.times
    psi_t = sol.psi_t
    dpsi_t = sol.dpsi_dt
    w_t = sol.w_t
    d2psi_dt2 = sol.d2psi_dt2
    delta_primes = sol.delta_primes


    jorek_data = pd.read_csv(jorek_data_filename).fillna(0)
    jorek_times, jorek_flux = jorek_flux_at_q(jorek_data, q_profile, 2/1)
    

    min_time = 0.0#1e4
    max_time = 1e6

    # Model flux
    model_filt = ((times<max_time) & (times> min_time))
    times = times[model_filt]
    model_flux = psi_t[model_filt]
    model_flux_func = UnivariateSpline(times, model_flux, s=0)

    # JOREK flux
    jorek_filt = jorek_times>min_time
    jorek_times = jorek_times[jorek_filt]
    jorek_flux = jorek_flux[jorek_filt]

    # Parabolic fitting
    min_t2_time = 1.9e5
    max_t2_time = 5e5
    c_0, c_1, c_2 = get_parab_coefs(params, model_flux_func(min_t2_time))

    t = np.linspace(min_t2_time, max_t2_time, 100)

    fig, ax = plt.subplots(1, figsize=(5,4))

    ax.plot(times, model_flux, label='Model', color='black')
    ax.plot(jorek_times, jorek_flux, label='JOREK', color='red')

    log_times = np.logspace(4, np.log(max(times)), 100)
    #ax.plot(log_times, 1.0/log_times, label='2/t dependence')


    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r"Time ($1/\omega_A$)")
    ax.set_ylabel(r"Flux at rational surface ($a^2 B_{\phi 0}$)")



    ax.plot(
        t, 
        c_0*(t-min_t2_time)**2 +c_1*(t-min_t2_time) + c_2, 
        color='green', linestyle='--', 
        label=r'$f(t) = at^2 + bt + c$' + \
            f"\n a={c_0:.2e},\n b={c_1:.2e},\n c={c_2:.2e}"
    )

    ax.legend()
    fig.tight_layout()
    
    savefig("flux_comparison_log_log")
    
    ax.set_xlim(left=40000, right=1e6)
    ax.set_ylim(bottom=1e-9, top=1e-2)
    fig.tight_layout()
    
    savefig("flux_comparison_log_log_zoom")
    
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    fig.set_size_inches(6, 4.65, forward=True)
    ax.set_xlim(left=0, right=4e5)
    ax.set_ylim(bottom=0, top=0.00175)
    fig.tight_layout()
    
    savefig("flux_comparison_lin_lin_early")
    
    ax.set_xlim(left=0, right=1e6)
    ax.set_ylim(bottom=0, top=0.00175)
    
    savefig("flux_comparison_lin_lin")
    
    
    ax.autoscale()
    ax.set_yscale("log")
    ax.set_xlim(left=0, right=5e5)
    savefig("flux_comparison_lin_log")


    plt.show()





if __name__=='__main__':
    ql_tm_vs_time()
    #check_model_t_dependence()
