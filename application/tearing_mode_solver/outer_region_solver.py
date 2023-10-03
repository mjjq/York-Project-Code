from scipy.integrate import odeint, solve_ivp
from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from scipy.interpolate import interp1d

from tearing_mode_solver.profiles import q, dj_dr, rational_surface
from tearing_mode_solver.helpers import savefig


def compute_derivatives(y: Tuple[float, float],
                        r: float,
                        poloidal_mode: int,
                        toroidal_mode: int,
                        j_profile_derivative,
                        q_profile,
                        axis_q: float,
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

@dataclass
class OuterRegionSolution():
    # Contains numerical solution to the outer solution differential
    # equation

    # All quantities in this class are normalised
    
    # Perturbed flux and derivative starting from the poloidal axis
    psi_forwards: np.array
    dpsi_dr_forwards: np.array
    dpsi_dr_f_func: callable
    
    # Radial domain for the forward solution
    r_range_fwd: np.array
    
    # Perturbed flux and derivative starting from the edge of the plasma
    # going inwards
    psi_backwards: np.array
    dpsi_dr_backwards: np.array
    dpsi_dr_b_func: callable
    
    # Radial domain for the backward solution
    r_range_bkwd: np.array
    
    # Location of the resonant surface
    r_s: float
    
def scale_tm_solution(tm: OuterRegionSolution, scale_factor: float)\
    -> OuterRegionSolution:
    return OuterRegionSolution(
        tm.psi_forwards*scale_factor, 
        tm.dpsi_dr_forwards*scale_factor, 
        tm.r_range_fwd, 
        tm.psi_backwards*scale_factor, 
        tm.dpsi_dr_backwards*scale_factor, 
        tm.r_range_bkwd, 
        tm.r_s
    )

def solve_system(poloidal_mode: int, 
                 toroidal_mode: int, 
                 axis_q: float,
                 resolution: float = 1e-6,
                 r_s_thickness: float = 1e-4) -> OuterRegionSolution:
    """
    Generate solution for peturbed flux over the minor radius of a cylindrical
    plasma given the mode numbers of the tearing mode.

    Parameters
    ----------
    poloidal_mode : int
        Poloidal mode number.
    toroidal_mode : int
        Toroidal mode number.
    axis_q : float, optional
        Value of the safety factor on-axis. The default is 1.0.
    n: int, optional
        Number of elements in the integrand each for the forwards and 
        backwards solutions. The default is 10000.

    Returns
    -------
    OuterRegionSolution:
        Quantities relating to the tearing mode solution.

    """
    initial_psi = 0.0
    initial_dpsi_dr = 1.0

    q_rs = poloidal_mode/(toroidal_mode*axis_q)
    if q_rs >= q(1.0) or q_rs <= q(0.0):
        raise ValueError("Rational surface located outside bounds")

    r_s = rational_surface(poloidal_mode/(toroidal_mode*axis_q))

    #r_s_thickness = 0.0001

    #print(f"Rational surface located at r={r_s:.4f}")

    # Solve from axis moving outwards towards rational surface
    r_range_fwd = np.arange(0.0, r_s-r_s_thickness, resolution)

    results_forwards = odeint(
        compute_derivatives,
        (initial_psi, initial_dpsi_dr),
        r_range_fwd,
        args = (poloidal_mode, toroidal_mode, dj_dr, q, axis_q),
        tcrit=(0.0)
    )

    psi_forwards, dpsi_dr_forwards = (
        results_forwards[:,0], results_forwards[:,1]
    )

    # Solve from minor radius moving inwards towards rational surface
    r_range_bkwd = np.arange(1.0, r_s+r_s_thickness, -resolution)

    results_backwards = odeint(
        compute_derivatives,
        (initial_psi, -initial_dpsi_dr),
        r_range_bkwd,
        args = (poloidal_mode, toroidal_mode, dj_dr, q, axis_q)
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
    dpsi_dr_forwards = dpsi_dr_forwards * bkwd_res_surface/fwd_res_surface
    
    #print(dpsi_dr_forwards)
    #print(r_range_fwd)
    
    # Recover original derivatives as the compute_derivatives function returns
    # r^2 * f' for r>0. For r=0, the compute_derivatives function returns f'
    # so no need to divide by r^2.
    dpsi_dr_forwards[1:] = dpsi_dr_forwards[1:]/(r_range_fwd[1:]**2)
    dpsi_dr_backwards = dpsi_dr_backwards/(r_range_bkwd**2)
    
    dpsi_dr_f_func = interp1d(
        r_range_fwd, 
        dpsi_dr_forwards, 
        fill_value=(dpsi_dr_forwards[0], dpsi_dr_forwards[-1]),
        bounds_error=False
    )
    dpsi_dr_b_func = interp1d(
        r_range_bkwd, 
        dpsi_dr_backwards,
        fill_value=(dpsi_dr_backwards[-1], dpsi_dr_backwards[0]),
        bounds_error=False
    )
    #print(dpsi_dr_forwards)
    
    return OuterRegionSolution(
        psi_forwards, dpsi_dr_forwards, dpsi_dr_f_func, r_range_fwd,
        psi_backwards, dpsi_dr_backwards, dpsi_dr_b_func, r_range_bkwd,
        r_s
    )
    
    # return psi_forwards, dpsi_dr_forwards, r_range_fwd, \
    #     psi_backwards, dpsi_dr_backwards, r_range_bkwd , r_s
    
    
def delta_prime(tm_sol: OuterRegionSolution,
                epsilon: float = 1e-10):
    """
    Calculate the discontinuity parameter close to the resonant surface for
    a numerical outer solution.

    Epsilon specifies the tolerance for the forward and backward solutions at
    the resonannt surface to be within. ValueError is raised if this check fails
    """
    psi_plus = tm_sol.psi_backwards[-1]
    psi_minus = tm_sol.psi_forwards[-1]
    
    if abs(psi_plus - psi_minus) > epsilon:
        raise ValueError(
            f"""Forwards and backward solutions 
            should be equal at resonant surface.
            (psi_plus={psi_plus}, psi_minus={psi_minus})."""
        )
    
    dpsi_dr_plus = tm_sol.dpsi_dr_backwards[-1]
    dpsi_dr_minus = tm_sol.dpsi_dr_forwards[-1]
    
    return (dpsi_dr_plus - dpsi_dr_minus)/psi_plus





def magnetic_shear(resonant_surface: float,
                   poloidal_mode: int,
                   toroidal_mode: int) -> float:
    """
    Calculate the magnetic shear of the plasma at the resonant surface using
    the globally defined q profile.

    s(r_s) = q'(r_s)/q(r_s)
    """
    r_s = resonant_surface
    
    r = np.linspace(0, 1, 1000)
    dr = r[1]-r[0]
    q_values = q(r)
    dq_dr = np.gradient(q_values, dr)
    rs_id = np.abs(r_s - r).argmin()
    
    m = poloidal_mode
    n = toroidal_mode
    
    return (m/n)*r_s*dq_dr[rs_id]

def gamma_constant() -> float:
    """
    Numerical solution to the integral int_{-\infty}^{\infty} (1+XY(X)) dX,
    where Y(X) = 0.5X\int_0^1 \exp(-0.5\mu X^2)/(1-\mu^2)^{1/4} d\mu
    """
    return 2.1236482729819393256107565

def growth_rate_scale(lundquist_number: float,
                      r_s: float,
                      poloidal_mode: float,
                      toroidal_mode: float) -> float:
    """
    Given some of the plasma paramters, calculate the value that multiplies
    by (Delta')^{4/5} to give the linear growth rate.

    I.e. \gamma/\omega_A = growth_rate_scale(...)*(Delta')^{4/5}
    """
   
    # Equivalent to 2*pi*Gamma(3/4)/Gamma(1/4)
    gamma_scale_factor = gamma_constant()
    
    m = poloidal_mode
    n = toroidal_mode
    S = lundquist_number
    
    s = magnetic_shear(r_s, m, n)
    
    grs = gamma_scale_factor**(-4/5)* r_s**(4/5) \
        * (n*s)**(2/5) / S**(3/5)
        
    return grs
    

def growth_rate(poloidal_mode: int,
                toroidal_mode: int,
                lundquist_number: float,
                axis_q: float = 1.0,
                resolution: float = 1e-6) -> Tuple[float, float]:
    """
    Calculate growth rate of a tearing mode in the linear regime.

    Returns a tuple containing the discontinuity parameter and the growth rate
    normalised to the Alfven frequency of the plasma.
    """
    
    m = poloidal_mode
    n = toroidal_mode
    S = lundquist_number
    
    try:
        tm = solve_system(m, n, axis_q, resolution)
    except ValueError:
        return np.nan, np.nan
    
    delta_p = delta_prime(tm)
    
    grs = growth_rate_scale(S, tm.r_s, m, n)

    growth_rate = grs*complex(delta_p)**(4/5)

    return delta_p, growth_rate.real

def layer_width(poloidal_mode: int,
                toroidal_mode: int,
                lundquist_number: float,
                axis_q: float = 1.0) -> float:
    """
    Calculate the thickness of the resistive layer in the linear regime.
    """
    m = poloidal_mode
    n = toroidal_mode
    S = lundquist_number

    try:
        tm = solve_system(m, n, axis_q)
    except ValueError:
        return np.nan

    delta_p, gr = growth_rate(m, n, S, axis_q)

    s = magnetic_shear(tm.r_s, m, n)

    return tm.r_s*(gr/(S*(n*s)**2))**(1/4)

def alfven_frequency_STEP():
    # Alfven frequency estimated for the STEP tokamak.
    # 1.35e6 Hz for STEP, see lab book "STEP parameters" log
    return 1.35e6

def ps_correction(alfven_freq: float,
                  poloidal_mode: int,
                  toroidal_mode: int):
    """
    Pfirsch schluter correction to the alfven frequency at
    resonant surface.
    """

    return alfven_freq / (1+2*poloidal_mode/toroidal_mode)

