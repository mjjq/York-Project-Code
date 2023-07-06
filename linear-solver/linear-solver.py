from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize_scalar
from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt

def dj_dr(radial_coordinate: float,
          shaping_exponent: float = 2.0) -> float:
    """
    Normalised derivative in the current profile.

    Current is normalised to the on-axis current J_0.
    radial_coordinate is normalised to the minor radius a.
    """
    r = radial_coordinate
    nu = shaping_exponent

    return -2*nu*(r) * (1-r**2)**(nu-1)

@np.vectorize
def q(radial_coordinate: float,
      shaping_exponent: float = 2.0) -> float:
    r = radial_coordinate
    nu = shaping_exponent
    
    # Prevent division by zero for small r values.
    # For this function, in the limit r->0, q(r)->1. This is proven
    # mathematically in the lab book.
    if np.abs(r) < 1e-10:
        return 1.0

    return (nu+1)*(r**2)/(1-(1-r**2)**(nu+1))

def rational_surface(target_q: float,
                     shaping_exponent: float = 2.0) -> float:
    """
    Compute the location of the rational surface of the q-profile defined in q().
    """
    
    # Establish the function to pass to scipy's scalar minimiser. We want to 
    # find the value of r such that q(r) = target_q. This is equivalent to
    # finding the value of r that minimises (q(r) - target_q)^2
    # Call vectorize so that it converts q(r) into a function that can take
    # np arrays. This is necessary because q(r) contains an if statement which
    # fails if the function is not vectorized.
    fun = np.vectorize(lambda r : (q(r, shaping_exponent) - target_q)**2)
    
    r = np.linspace(0,1,100)
    plt.plot(r, fun(r))
    
    print("Plotted results")
    
    rs = minimize_scalar(fun, bounds=(0.0, 1.0), method='bounded')
    print(rs)

    return rs.x


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

    initial_psi = 0.0
    initial_dpsi_dr = 1

    r_s = rational_surface(poloidal_mode/(toroidal_mode*axis_q))
    r_s_thickness = 0.01

    print(f"Rational surface located at r={r_s:.4f}")

    # Solve from axis moving outwards towards rational surface
    r_range_fwd = np.linspace(0.001, r_s-r_s_thickness, 10000)

    results_forwards = odeint(
        compute_derivatives,
        (initial_psi, initial_dpsi_dr),
        r_range_fwd,
        args = (poloidal_mode, toroidal_mode, dj_dr, q)
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
        args = (poloidal_mode, toroidal_mode, dj_dr, q)
    )

    psi_backwards, dpsi_dr_backwards = (
        results_backwards[:,0], results_backwards[:,1]
    )
    #print(psi_backwards)
    #print(dpsi_dr_backwards)

    fig, axs = plt.subplots(3, figsize=(6,10), sharex=True)
    ax, ax2, ax3 = axs

    ax.plot(r_range_fwd, psi_forwards, label='Solution below $\hat{r}_s$')
    ax.plot(r_range_bkwd, psi_backwards, label='Solution above $\hat{r}_s$')
    rs_line = ax.vlines(
        r_s, ymin=0.0, ymax=np.max([psi_forwards, psi_backwards]),
        linestyle='--', color='red', 
        label=f'Rational surface $\hat{{q}}(\hat{{r}}_s) = {poloidal_mode}/{toroidal_mode}$'
    )

    ax3.set_xlabel("Normalised minor radial co-ordinate (r/a)")
    ax.set_ylabel("Normalised perturbed flux ($\delta \psi / a^2 J_\phi$)")
    
    r = np.linspace(0, 1.0, 100)
    ax2.plot(r, dj_dr(r))
    ax2.set_ylabel("Normalised current gradient $[d\hat{J}_\phi/d\hat{r}]$")
    ax2.vlines(
        r_s, 
        ymin=np.min(dj_dr(r)), 
        ymax=np.max(dj_dr(r)), 
        linestyle='--', 
        color='red'    
    )
    
    ax3.plot(r, np.vectorize(q)(r))
    ax3.set_ylabel("Normalised q-profile $[\hat{q}(\hat{r})]$")
    ax3.vlines(
        r_s, 
        ymin=np.min(q(r)), 
        ymax=np.max(q(r)), 
        linestyle='--', 
        color='red'    
    )
    #ax3.set_yscale('log')
    ax.legend()
    


if __name__=='__main__':
    solve_system()
    plt.tight_layout()
    plt.savefig("tm-with-q-djdr.png", dpi=300)
    plt.show()
