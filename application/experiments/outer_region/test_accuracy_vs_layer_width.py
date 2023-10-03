from matplotlib import pyplot as plt

import imports
from tearing_mode_solver.outer_region_solver import *

def test_accuracy_vs_layer_width():
    """
    Calculate the discontinuity parameter for a mode using outer tearing mode
    solutions of different resolutions.

    We then plot the discontinuity parameter (Delta') as a function of
    resolution to understand how the error in Delta' changes.

    We can use the output of this test to determine a reasonable resolution for
    the precision we need.
    """
    m=3
    n=2
    lundquist_number = 1e8

    resolution = 1e-4
    layer_widths = np.logspace(-7, -2, 10)

    delta_ps = []

    for width in layer_widths:
        tm = solve_system(m, n, 1.0, resolution, width)
        delta_p = delta_prime(tm)

        delta_ps.append(delta_p)

    fig, ax = plt.subplots(1, figsize=(4,3))
    print(layer_widths)
    print(delta_ps)
    ax.plot(layer_widths, delta_ps, color='black')
    ax.grid(which='major')

    ax.set_xscale('log')

    ax.set_xlabel(r"Normalised layer width $\hat{r}_s \delta$")
    ax.set_ylabel(r"$a\Delta'$")
    fig.tight_layout()

    savefig(f"accuracy_vs_layer_width_(m,n)={m},{n}_res={resolution:.2e}")
    plt.show()

if __name__=='__main__':
    test_accuracy_vs_layer_width()
