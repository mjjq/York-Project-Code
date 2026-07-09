import numpy as np
from scipy.special import j0, j1, jn_zeros

def solve_diffusion(r: np.array,
                    t: float,
                    B_init: float,
                    B_applied: float,
                    n_harmonics: int = 10000) -> np.array:
    """
    Solve resistive diffusion of toroidal field in
    a cylinder assuming constant resistivity.

    Use r, theta, z co-ordinates.

    :param r: Radial co-ordinate normalised to minor radius a
    :param t: Time normalised to resistive timescale
    :param B_init: Initial uniform plasma toroidal field
    :param B_applied: Toroidal field applied to edge of plasma
    :param n_harmonics: Number of Bessel harmonics used in solution
    """
    zeros = jn_zeros(0, n_harmonics)
    j0_vals = j0(np.outer(zeros,r))
    j1_vals = j1(zeros)
    exp_decay = np.exp(-zeros**2 * t)

    c_n_vals = 2.0*exp_decay/(j1_vals*zeros)

    fourier_args = np.vecmat(c_n_vals, j0_vals)

    return B_init + (B_applied-B_init)*(1.0-fourier_args)

if __name__=='__main__':
    r = np.linspace(0.0, 1.0, 100)
    times = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]

    from matplotlib import pyplot as plt

    for t in times:
        sol = solve_diffusion(
            r, t, 0.0, 0.5
        )

        plt.plot(
            r, sol, label=f"t={t:.2g}"
        )

    plt.legend()
    plt.show()