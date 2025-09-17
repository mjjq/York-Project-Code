from matplotlib import pyplot as plt
import numpy as np

from tearing_mode_solver.helpers import TimeDependentSolution, TearingModeParameters
from tearing_mode_solver.conversions import solution_time_scale, time_unit_label
from tearing_mode_solver.outer_region_solver import rational_surface

def island_width_plot(params: TearingModeParameters,
                      ql_solution: TimeDependentSolution,
                      si_units: bool = False):
    fig, ax_w = plt.subplots(1, figsize=(4,3))

    scaled_solution = solution_time_scale(params, ql_solution, si_units)
    unit_label = time_unit_label(si_units)

    ax_w.plot(scaled_solution.times, scaled_solution.w_t, color='black')
    ax_w.set_xlabel(f"Time ({unit_label})")
    ax_w.set_ylabel("w/a")

    ax_w.grid()

    fig.tight_layout()

def phase_plot(params: TearingModeParameters,
               ql_solution: TimeDependentSolution, 
               si_units: bool = False):
    fig, ax_phase = plt.subplots(1, figsize=(4,3))

    scaled_solution = solution_time_scale(params, ql_solution, si_units)
    unit_label = time_unit_label(si_units)

    dw_vec = np.diff(scaled_solution.w_t)
    dt_vec = np.diff(scaled_solution.times)

    dwdt = dw_vec/dt_vec

    ax_phase.plot(scaled_solution.w_t[:-1], dwdt, color='black')
    ax_phase.set_xlabel("w/a")
    ax_phase.set_ylabel(f"dw/dt (a/{unit_label})")

    ax_phase.set_yscale('log')

    ax_phase.grid()

    fig.tight_layout()

def delta_prime_plot(params: TearingModeParameters,
                     ql_solution: TimeDependentSolution,
                     si_units: bool = False):
    fig, ax = plt.subplots(1, figsize=(4,3))
    ax.grid()

    scaled_solution = solution_time_scale(params, ql_solution, si_units)
    unit_label = time_unit_label(si_units)

    r_s = rational_surface(
        params.q_profile, 
        params.poloidal_mode_number/params.toroidal_mode_number
    )

    ax.plot(scaled_solution.times, r_s*scaled_solution.delta_primes, color='black')

    ax.set_xlabel(f"Time ({unit_label})")
    ax.set_ylabel("$r_s \Delta'(w)$")
    fig.tight_layout()

def delta_prime_phase_plot(params: TearingModeParameters,
                           ql_solution: TimeDependentSolution,
                           si_units: bool = False):
    fig, ax = plt.subplots(1, figsize=(4,3))
    ax.grid()

    scaled_solution = solution_time_scale(params, ql_solution, si_units)
    unit_label = time_unit_label(si_units)

    r_s = rational_surface(
        params.q_profile, 
        params.poloidal_mode_number/params.toroidal_mode_number
    )

    ax.plot(
        scaled_solution.w_t/r_s, 
        r_s*scaled_solution.delta_primes, 
        color='black'
    )

    ax.set_xlabel(f"$w/r_s$")
    ax.set_ylabel("$r_s \Delta'(w)$")
    fig.tight_layout()