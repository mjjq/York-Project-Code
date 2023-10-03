from matplotlib import pyplot as plt

import imports
from tearing_mode_solver.outer_region_solver import *

def growth_rate_vs_mode_number():
    """
    Generate a table of data of the growth rate for different tearing modes
    in the linear regime.

    Normalised growth rates are calculated in addition to absolute growth rates
    for STEP parameters.
    """
    modes = [
        (2,1),
        (3,2),
        (4,2),
        (4,3),
        (5,2),
        (5,3),
        (5,4),
        (6,3),
        (6,4),
        (6,5)
    ]
    lundquist_number = 1e8

    fig, ax = plt.subplots(1)

    results = []

    alfven_frequency = alfven_frequency_STEP()

    for m,n in modes:
        delta_p, gr = growth_rate(m,n,lundquist_number)

        alfven_corrected = ps_correction(alfven_frequency, m, n)

        abs_growth = gr*alfven_corrected

        linear_layer_width = layer_width(m,n,lundquist_number)

        results.append(
            (delta_p, gr, alfven_corrected, abs_growth, linear_layer_width)
        )

    print(f"mode & delta_p & gamma/omega & omega_bar & gamma & lin_growth")
    for i,mode in enumerate(modes):
        m, n = mode
        (delta_p, growth, af_corrected,
         abs_growth, linear_layer_width) = results[i]
        print(
            f"{m} & {n} & {delta_p:.2f} & ${growth:.2e}$ & "
            f"${af_corrected:.2e}$ & ${abs_growth:.2e}$ & "
            f"${linear_layer_width:.2e}$"+r"\\")

if __name__=='__main__':
    growth_rate_vs_mode_number()
