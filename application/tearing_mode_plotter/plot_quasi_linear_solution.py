from matplotlib import pyplot as plt
import numpy as np

from tearing_mode_solver.helpers import TimeDependentSolution, savefig, TearingModeParameters
from tearing_mode_solver.conversions import time_unit_label, solution_time_scale

def plot_perturbed_flux(params: TearingModeParameters,
                        ql_solution: TimeDependentSolution, 
                        si_units: bool = False):
    fig, ax = plt.subplots(1, figsize=(4, 3))

    scaled_solution = solution_time_scale(params, ql_solution, si_units)
    unit_label = time_unit_label(si_units)

    ax.plot(scaled_solution.times, scaled_solution.psi_t, label="Flux", color="black")

    ax.set_xlabel(f"Time [{unit_label}]")
    ax.set_ylabel(r"Normalised $\delta\psi$ (arb)")

    # ax.set_yscale('log')
    # ax2.set_yscale('log')
    ax.set_xscale("linear")
    ax.set_yscale("log")

    ax.grid()

    fig.tight_layout()

    savefig("perturbed_flux")

def plot_growth_rate(params: TearingModeParameters,
                     ql_solution: TimeDependentSolution,
                     si_units: bool = False):
    fig_g, ax_g = plt.subplots(1, figsize=(4,3))

    scaled_solution = solution_time_scale(params, ql_solution, si_units)
    unit_label = time_unit_label(si_units)

    growth_rates = scaled_solution.dpsi_dt/scaled_solution.psi_t
    ax_g.plot(scaled_solution.times, growth_rates, color='black')

    ax_g.grid()
    ax_g.set_xlabel(f"Time [{unit_label}]")

    ax_g.set_ylabel("Growth rate "f"[1/{unit_label}]")

    fig_g.tight_layout()

    savefig(f"growth_rate")