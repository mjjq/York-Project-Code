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

def d2j_dr2(radial_coordinate: float,
            shaping_exponent: float = 2.0) -> float:
    r = radial_coordinate
    nu = shaping_exponent
    
    return -2*nu*(1-r**2)**(nu-1) + 4*nu*(r**2)*(1-r**2)**(nu-2)

@np.vectorize
def q(radial_coordinate: float,
      shaping_exponent: float = 2.0) -> float:
    r = radial_coordinate
    nu = shaping_exponent
    
    # Prevent division by zero for small r values.
    # For this function, in the limit r->0, q(r)->1. This is proven
    # mathematically in the lab book.
    #print(r)
    if np.abs(r) < 1e-5:
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
    
    rs = minimize_scalar(fun, bounds=(0.0, 1.0), method='bounded')

    return rs.x


def compute_derivatives(y: Tuple[float, float],
                        r: float,
                        poloidal_mode: int,
                        toroidal_mode: int,
                        j_profile_derivative,
                        j_profile_second_derivative,
                        q_profile,
                        axis_q: float = 1.0,
                        epsilon: float = 1e-5) -> Tuple[float, float]:
    """
    Compute derivatives for the perturbed flux outside of the resonant region
    (resistivity and inertia neglected).
    
    The equation we are solving is
    
    r^2 f'' + rf' + 2rfA - m^2 f = 0,
    
    where f is the normalised perturbed flux, r is the radial co-ordinate,
    m is the poloidal mode number, and A = (q*m*dj_dr)/(n*q_0*q - m), where
    q is the normalised safety factor, dj_dr is the normalised gradient in
    the toroidal current, q_0 is the normalised safety factor at r=0, and
    n is the toroidal mode number.
    
    The equation is singular at r=0, so to get ODEINT to play nicely we
    formulate the following coupled ODE:
        
    y = [f, r^2 f']
         
    y'= [f', r^2 f'' + 2rf']
    
      = [f', f(m^2 - 2rA) + rf']
      
    Then, for r=0, 
    
    y'= [f', m^2 f]

    We set the initial f' at r=0 to some arbitrary positive value. Note that,
    for r>0, this function will then receive a tuple containing f and 
    **r^2 f'** instead of just f and f', as this is what we are calculating
    in y as defined above.
    
    Hence, for r>0, we must divide the incoming f' by r^2.

    Parameters
    ----------
    y : Tuple[float, float]
        Perturbed flux and radial derivative in perturbed flux.
    r : float
        Radial co-ordinate.
    poloidal_mode : int
        Poloidal mode number.
    toroidal_mode : int
        Toroidal mode number.
    j_profile_derivative : func
        Derivative in the current profile. Must be a function which accepts
        the radial co-ordinate r as a parameter.
    j_profile_second_derivative : f
        DESCRIPTION.
    q_profile : TYPE
        Safety factor profile. Must be a function which accepts the radial
        co-ordinate r as a parameter.
    axis_q : float, optional
        Value of the normalised safety factor on-axis. The default is 1.0.
    epsilon : float, optional
        Tolerance value to determine values of r which are sufficiently close
        to r=0. The default is 1e-5.

    Returns
    -------
    Tuple[float, float]
        Radial derivative in perturbed flux and second radial derivative in 
        perturbed flux.

    """
    psi, dpsi_dr = y
    
    if np.abs(r) > epsilon:
        dpsi_dr = dpsi_dr/r**2

    m = poloidal_mode
    n = toroidal_mode
    q_0 = axis_q
    
    if np.abs(r) < epsilon:
        d2psi_dr2 = psi*m**2
    else:
        dj_dr = j_profile_derivative(r)
        q = q_profile(r)
        A = (q*m*dj_dr)/(n*q_0*q - m)
        d2psi_dr2 = psi*(m**2 - 2*A*r) + r*dpsi_dr
        
    #print(dpsi_dr)

    return dpsi_dr, d2psi_dr2


def solve_system():
    poloidal_mode = 2
    toroidal_mode = 1
    axis_q = 1.0

    initial_psi = 0.0
    initial_dpsi_dr = 1.0

    r_s = rational_surface(poloidal_mode/(toroidal_mode*axis_q))
    r_s_thickness = 0.0001

    print(f"Rational surface located at r={r_s:.4f}")

    # Solve from axis moving outwards towards rational surface
    r_range_fwd = np.linspace(0.01, r_s-r_s_thickness, 10000)

    results_forwards = odeint(
        compute_derivatives,
        (initial_psi, initial_dpsi_dr),
        r_range_fwd,
        args = (poloidal_mode, toroidal_mode, dj_dr, d2j_dr2, q),
        tcrit=(0.0)
    )

    psi_forwards, dpsi_dr_forwards = (
        results_forwards[:,0], results_forwards[:,1]
    )

    # Solve from minor radius moving inwards towards rational surface
    r_range_bkwd = np.linspace(1.0, r_s+r_s_thickness, 10000)

    results_backwards = odeint(
        compute_derivatives,
        (initial_psi, -initial_dpsi_dr),
        r_range_bkwd,
        args = (poloidal_mode, toroidal_mode, dj_dr, d2j_dr2, q)
    )

    psi_backwards, dpsi_dr_backwards = (
        results_backwards[:,0], results_backwards[:,1]
    )
    #print(psi_backwards)
    #print(dpsi_dr_backwards)
    
    # Rescale the forwards solution such that its value at the resonant
    # surface matches the psi of the backwards solution. This is equivalent
    # to fixing the initial values of the derivatives such that the above
    # relation is satisfied
    fwd_res_surface = psi_forwards[-1]
    bkwd_res_surface = psi_backwards[-1]
    psi_forwards = psi_forwards * bkwd_res_surface/fwd_res_surface

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
    ax.legend(prop={'size': 8})
    


if __name__=='__main__':
    solve_system()
    plt.tight_layout()
    plt.savefig("tm-with-q-djdr.png", dpi=300)
    plt.show()
