from matplotlib import pyplot as plt
import numpy as np
from typing import List

from tearing_mode_solver.outer_region_solver import solve_system, TearingModeParameters, delta_prime, OuterRegionSolution
from tearing_mode_solver.profiles import generate_q_profile, generate_j_profile

from jorek_tools.quasi_linear_model.get_tm_parameters import get_parameters

def test_accuracy_vs_layer_width():
    """
    Calculate the discontinuity parameter for a mode using outer tearing mode
    solutions of different resolutions.

    We then plot the discontinuity parameter (Delta') as a function of
    resolution to understand how the error in Delta' changes.

    We can use the output of this test to determine a reasonable resolution for
    the precision we need.
    """
    delta_ps = []
    sols: List[OuterRegionSolution] = []

    current_scale_factors = np.linspace(1.0, 0.1, 5)

    for j_scale in current_scale_factors:
        q_profile = generate_q_profile(1.0, 2.0)
        j_profile = generate_j_profile(1.0, 2.0)

        rs, js = zip(*j_profile)
        js = j_scale*np.array(js)

        j_profile = zip(rs, js)

        params = TearingModeParameters(
            poloidal_mode_number=2,
            toroidal_mode_number=1,
            lundquist_number=1e8,
            initial_flux=1e-12,
            B0=j_scale,
            R0=20.0,
            q_profile=q_profile,
            j_profile=j_profile
        )


        tm = solve_system(params)
        delta_p = delta_prime(tm)

        delta_ps.append(delta_p)
        sols.append(tm)

    fig, ax = plt.subplots(1, figsize=(4,3))
    ax.scatter(current_scale_factors, delta_ps, color='black')
    ax.grid(which='major')

    ax.set_xscale('log')

    ax.set_xlabel(r"J profile scale factor")
    ax.set_ylabel(r"$a\Delta'$")
    fig.tight_layout()

    plt.show()

if __name__=='__main__':
    test_accuracy_vs_layer_width()
