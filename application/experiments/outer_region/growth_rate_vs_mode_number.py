from matplotlib import pyplot as plt
import sys

import imports
from tearing_mode_solver.outer_region_solver import *
from tearing_mode_solver.profiles import generate_j_profile, generate_q_profile

def growth_rate_vs_mode_number():
    """
    Generate a table of data of the growth rate for different tearing modes
    in the linear regime.

    Normalised growth rates are calculated in addition to absolute growth rates
    for STEP parameters.
    """
    modes = [
        (2,1),
        (3,1),
        (4,1)
    ]
    lundquist_number = 1e8

    fig, ax = plt.subplots(1)

    results = []

    alfven_frequency = alfven_frequency_STEP()

    q_prof = generate_q_profile(1.0, 2.0)
    j_prof = generate_j_profile(2.0)

    

    for m,n in modes:
        try:
            params = TearingModeParameters(
                poloidal_mode_number=m,
                toroidal_mode_number=n,
                lundquist_number=lundquist_number,
                initial_flux=1e-12,
                B0=1,
                R0=40,
                q_profile=q_prof,
                j_profile=j_prof
            )
            outer_sol = solve_system(params)

            r_s = rational_surface(q_prof, m/n)

            delta_p, gr = growth_rate(m, n, lundquist_number, q_prof, outer_sol)

            alfven_corrected = ps_correction(alfven_frequency, m, n)

            abs_growth = gr*alfven_corrected

            #linear_layer_width = layer_width(m,n,lundquist_number, q_prof)

            results.append(
                (m, n, delta_p, r_s*delta_p, gr, alfven_corrected, abs_growth)
            )
        except ValueError as e:
            print(e, file=sys.stderr)

    print(f"\# m n Delta' r_s*Delta' gamma/omega omega_bar gamma")
    for res in results:
        (m, n, delta_p, r_sdelta_p, growth, af_corrected,
         abs_growth) = res
        print(
            f"{m} {n} {delta_p:.2f} {r_sdelta_p:.2f} {growth:.2e} "
            f"{af_corrected:.2e} ${abs_growth:.2e}$"
        )

if __name__=='__main__':
    growth_rate_vs_mode_number()
