import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt

from tearing_mode_solver.profiles import generate_j_profile, generate_q_profile
from tearing_mode_solver.delta_model_solver import solve_time_dependent_system
from tearing_mode_solver.outer_region_solver import (
    solve_system,
    OuterRegionSolution,
    growth_rate,
    delta_prime_non_linear
)
from tearing_mode_solver.helpers import (
    TearingModeParameters,
    sim_to_disk,
    TimeDependentSolution,
)
from tearing_mode_plotter.plot_magnetic_island_width import phase_plot, island_width_plot
from tearing_mode_plotter.plot_quasi_linear_solution import plot_perturbed_flux, plot_growth_rate
from jorek_tools.quasi_linear_model.central_density_si import central_density_si

def main():
    lundquist_numbers = np.logspace(4, 9, 6)
    sols = []

    fig, ax = plt.subplots(1)
    ax.set_yscale('log')

    for lundquist_number in lundquist_numbers:
        q_profile = generate_q_profile(axis_q=1.0, shaping_exponent=2.0)
        j_profile = generate_j_profile(axis_q=1.0, shaping_exponent=2.0)

        poloidal_mode_number = 2
        toroidal_mode_number = 1
        init_flux = 1e-12 # JOREK flux at which the simulation numerically stabilises
        t0 = 0  # This is the jorek time at which the simulation numerically stabilises
        nsteps = 1000

        print(f"lundquist number: {lundquist_number:.2g}")

        rho0 = 1.0
        B_tor = 1.0
        R_0 = 1.0

        params = TearingModeParameters(
            poloidal_mode_number=poloidal_mode_number,
            toroidal_mode_number=toroidal_mode_number,
            lundquist_number=lundquist_number,
            initial_flux=init_flux,
            B0=B_tor,
            R0=R_0,
            q_profile=q_profile,
            j_profile=j_profile,
            rho0=rho0
        )

        # Typically, several lundquist numbers needed to reach saturation
        t1 = t0 + 50*lundquist_number**(3/5)

        times = np.linspace(t0, t1, nsteps)
        #print(max(times))

        outer_sol = solve_system(params)
        delta_p, gr = growth_rate(
            params.poloidal_mode_number,
            params.toroidal_mode_number,
            lundquist_number,
            q_profile,
            outer_sol
        )

        print(f"Delta'(0)={delta_p:.2g}")

        ql_solution = solve_time_dependent_system(params, times)
        
        ax.plot(
            ql_solution.times*lundquist_number**(-3/5), ql_solution.psi_t,
            label=f"S={lundquist_number:.2g}"
        )

    ax.legend()
    ax.set_xlabel(r"Time ($S^{-3/5}\tau_A$)")
    ax.set_ylabel(r"$\delta\psi$ (arb)")
    plt.show()

if __name__=='__main__':
    main()