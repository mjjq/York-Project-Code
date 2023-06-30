from scipy.integrate import odeint
from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt

def dj_dr(radial_coordinate: float,
          shaping_exponent: float = 1.0) -> float:
    """
    Normalised derivative in the current profile.

    Current is normalised to the on-axis current J_0.
    radial_coordinate is normalised to the minor radius a.
    """
    r = radial_coordinate
    nu = shaping_exponent

    return -2*nu*(r**2) * (1-r**2)**(nu-1)

def q(radial_coordinate: float,
      shaping_exponent: float = 1.0) -> float:
    r = radial_coordinate
    nu = shaping_exponent

    return (1-r**2)**(-nu)

def rational_surface(target_q: float,
                     shaping_exponent: float = 1.0) -> float:
    """
    Compute the location of the rational surface of the q-profile defined in q().
    """

    return np.sqrt(1-target_q**(-1/shaping_exponent))


def compute_derivatives(y: Tuple[float, float],
                        r: float,
                        poloidal_mode: int,
                        toroidal_mode: int,
                        j_profile_derivative,
                        q_profile,
                        axis_q: float = 1.0) -> Tuple[float, float]:
    psi, dpsi_dr = y

    m = poloidal_mode
    n = toroidal_mode
    q = q_profile(r)
    q_0 = axis_q
    dj_dr = j_profile_derivative(r)

    d2psi_dr2 = -dpsi_dr/r**2 + psi * (
            (m/r)**2 - 2.0*q*m/(r*(n*q*q_0-m))*dj_dr
        )

    return dpsi_dr, d2psi_dr2


def solve_system():
    poloidal_mode = 2
    toroidal_mode = 1
    axis_q = 1.0

    initial_psi = 0
    initial_dpsi_dr = 100

    r_s = rational_surface(poloidal_mode/(toroidal_mode*axis_q))
    r_s_thickness = 0.001

    print(f"Rational surface located at r={r_s:.4f}")

    # Solve from axis moving outwards towards rational surface
    r_range_fwd = np.linspace(0.001, r_s-r_s_thickness, 10000)

    results_forwards = odeint(
        compute_derivatives,
        (initial_psi, initial_dpsi_dr),
        r_range_fwd,
        args = (poloidal_mode, toroidal_mode, dj_dr, q),
        tcrit = [0.0]
    )

    psi_forwards, dpsi_dr_forwards = (
        results_forwards[:,0], results_forwards[:,1]
    )

    # Solve from minor radius moving inwards towards rational surface
    r_range_bkwd = np.linspace(0.999, r_s+r_s_thickness, 10000)

    results_backwards = odeint(
        compute_derivatives,
        (initial_psi, -initial_dpsi_dr),
        r_range_bkwd,
        args = (poloidal_mode, toroidal_mode, dj_dr, q),
        tcrit = [1.0]
    )

    psi_backwards, dpsi_dr_backwards = (
        results_backwards[:,0], results_backwards[:,1]
    )

    fig, ax = plt.subplots(1)

    ax.plot(r_range_fwd, psi_forwards)
    ax.plot(r_range_bkwd, psi_backwards)

    ax.set_xlabel("Normalised minor radial co-ordinate (r/a)")
    ax.set_ylabel("Normalised perturbed flux ($\delta \psi / a^2 J_\phi$)")


if __name__=='__main__':
    solve_system()
    plt.show()
