import numpy as np
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
from experiments.jorek_growth_comparison.linear_nonlinear_comparison import plot_against_backward_rutherford

def main():
    lundquist_numbers = [1e4, 1e6, 1e8, 1e9]
    sols = []

    for lundquist_number in lundquist_numbers:
        q_profile = generate_q_profile(axis_q=1.0, shaping_exponent=2.0)
        j_profile = generate_j_profile(axis_q=1.0, shaping_exponent=2.0)

        poloidal_mode_number = 2
        toroidal_mode_number = 1
        init_flux = 1e-12 # JOREK flux at which the simulation numerically stabilises
        t0 = 0  # This is the jorek time at which the simulation numerically stabilises
        nsteps = int(10000*(lundquist_number/lundquist_numbers[0])**(1/5))

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
        t1 = t0 + 0.5*lundquist_number

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
        sim_to_disk(f"ql_S{lundquist_number:.1g}", params, ql_solution)

        plot_against_backward_rutherford(params, ql_solution)

    plt.show()

if __name__=='__main__':
    main()