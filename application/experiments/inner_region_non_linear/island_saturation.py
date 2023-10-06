# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from linear_solver import OuterRegionSolution, solve_system




def island_saturation():
    """
    Plot delta' as a function of the magnetic island width using the non-linear
    rutherford equation.

    Returns
    -------
    None.

    """
    poloidal_mode = 2
    toroidal_mode = 1
    m = poloidal_mode
    n = toroidal_mode
    axis_q = 1.0
    
    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)
    
    island_widths = np.linspace(0.0, 1.0, 100)
    
    delta_ps = [delta_prime_non_linear(tm, w) for w in island_widths]
    
    fig, ax = plt.subplots(1, figsize=(4,3))
    
    ax.plot(island_widths, delta_ps, label=f"(m,n)=({m},{n})", color='black')
    
    ax.set_xlabel(r"Normalised island width ($a$)")
    ax.set_ylabel(r"$a\Delta ' (w/a)$")
    
    ax.hlines(
        0.0, xmin=0.0, xmax=1.0, color='red',
        linestyle='--', label=r"Marginal stability"
    )
    fig.tight_layout()

    ax.legend()

    #plt.show()
    plt.savefig(f"./output/island_saturation_(m,n)=({m},{n}).png", dpi=300)

    
if __name__=='__main__':
    island_saturation()
