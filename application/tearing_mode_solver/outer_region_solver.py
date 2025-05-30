from scipy.integrate import odeint, solve_ivp
from typing import Tuple, List
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from scipy.interpolate import interp1d, UnivariateSpline

try:
	from scipy.integrate import simpson
except ImportError:
	from scipy.integrate import simps as simpson

from tearing_mode_solver.profiles import (
    rational_surface, magnetic_shear_profile, magnetic_shear
)
from tearing_mode_solver.helpers import savefig, TearingModeParameters


def compute_derivatives(y: Tuple[float, float],
                        r: float,
                        poloidal_mode: int,
                        toroidal_mode: int,
                        B0: float,
                        R0: float,
                        q_profile: callable,
                        dj_dr_profile: callable,
                        epsilon: float = 1e-6) -> Tuple[float, float]:
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
    B0: float
        On-axis toroidal field in Tesla
    R0: float
        Major radius of the tokamak
    q_profile : TYPE
        Safety factor profile. Must be a function which accepts the radial
        co-ordinate r as a parameter.
    dj_dr_profile : func
        Derivative in the current profile. Must be a function which accepts
        the radial co-ordinate r as a parameter.
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
    
    if np.abs(r) < epsilon:
        d2psi_dr2 = psi*m**2
    else:
        dj_dr = dj_dr_profile(r)
        q = q_profile(r)
        A = (q*m*dj_dr)/(n*q - m)
        d2psi_dr2 = psi*(m**2 - A*r) + r*dpsi_dr
        
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

def solve_system(params: TearingModeParameters,
                 resolution: float = 1e-6,
                 r_s_thickness: float = 1e-4) -> OuterRegionSolution:
    """
    Generate solution for peturbed flux over the minor radius of a cylindrical
    plasma given the mode numbers of the tearing mode.

    Parameters
    ----------
    tm: TearingModeParameters
    
    Tearing mode input parameters

    Returns
    -------
    OuterRegionSolution:
        Quantities relating to the tearing mode solution.

    """
    initial_psi = 0.0
    initial_dpsi_dr = 1.0

    q_profile = params.q_profile
    j_profile = params.j_profile
    poloidal_mode = params.poloidal_mode_number
    toroidal_mode = params.toroidal_mode_number
    B0 = params.B0
    R0 = params.R0

    r_vals, q_vals = zip(*q_profile)
    q_func = UnivariateSpline(r_vals, q_vals, s=0.0)
    
    r_vals, j_vals = zip(*j_profile)
    j_func = UnivariateSpline(r_vals, j_vals, s=0.0)
    dj_dr_func = j_func.derivative()

    q_rs = poloidal_mode/(toroidal_mode)
    if q_rs >= q_func(1.0) or q_rs <= q_func(0.0):
        raise ValueError("Rational surface located outside bounds")

    r_s = rational_surface(q_profile, poloidal_mode/toroidal_mode)
    
    
    #r_s_thickness = 0.0001

    #print(f"Rational surface located at r={r_s:.4f}")

    # Solve from axis moving outwards towards rational surface
    r_range_fwd = np.arange(0.0, r_s-r_s_thickness, resolution)

    results_forwards = odeint(
        compute_derivatives,
        (initial_psi, initial_dpsi_dr),
        r_range_fwd,
        args = (
            poloidal_mode, toroidal_mode, B0, R0, q_func, dj_dr_func
        ),
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
        args = (
            poloidal_mode, toroidal_mode, B0, R0, q_func, dj_dr_func
        )
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



def gamma_constant() -> float:
    """
    Numerical solution to the integral int_{-\infty}^{\infty} (1+XY(X)) dX,
    where Y(X) = 0.5X\int_0^1 \exp(-0.5\mu X^2)/(1-\mu^2)^{1/4} d\mu
    """
    return 2.1236482729819393256107565
    

def growth_rate_full(poloidal_mode: int,
                     toroidal_mode: int,
                     lundquist_number: float,
                     r_s: float,
                     mag_shear: float,
                     delta_p: float) -> float:
    """
    Calculate the growth rate of a tearing mode in the linear regime.

    :param poloidal_mode: Poloidal mode number
    :param toroidal_mode: Toroidal mode number
    :param lundquist_number: Lundquist number (tau_R/tau_A)
    :param r_s: Radius of rational surface (normalised to a)
    :param mag_shear: Magnetic shear
    :param delta_p: Delta' (normalised to a, i.e. a*Delta'[SI])

    :return: Growth rate normalised to Alfven frequency, gamma/omega_A
    """
    m = poloidal_mode
    n = toroidal_mode
    s = mag_shear
    S = lundquist_number
    
    ps_corr = (1+2*(m/n)**2)

    gamma_scale_factor = gamma_constant()

    grs = gamma_scale_factor**(-4/5)* r_s**(4/5) \
        * (n*s)**(2/5) / S**(3/5) / ps_corr**(1/5)

    growth_rate = grs*complex(delta_p)**(4/5)

    return growth_rate.real


def growth_rate(poloidal_mode: int,
                toroidal_mode: int,
                lundquist_number: float,
                q_profile: List[Tuple[float, float]],
                tm: OuterRegionSolution) -> Tuple[float, float]:
    """
    Calculate growth rate of a tearing mode in the linear regime.

    Returns a tuple containing the discontinuity parameter and the growth rate
    normalised to the Alfven frequency of the plasma.
    """
    
    m = poloidal_mode
    n = toroidal_mode
    S = lundquist_number
    
    delta_p = delta_prime(tm)
    mag_shear = magnetic_shear(q_profile, tm.r_s)
    
    gr = growth_rate_full(
        m, n, S, tm.r_s, mag_shear, delta_p 
    )

    return delta_p, gr




def diffusion_width(chi_perp: float,
                    chi_parallel: float,
                    r_s: float,
                    aspect_ratio: float,
                    toroidal_mode_number: int,
                    magnetic_shear: float) -> float:
    """
    Calculate diffusion length scale as per Lutjens 2001, pp. 4268.
    We normalise the length scale to the plasma minor radius a.

    :param chi_perp: Perpendicular diffusion coefficient (arb units,
        must be identical to chi_parallel units)
    :param chi_parallel: Parallel diffusion coefficent (arb units)
    :param r_s: Normalised radius of rational surface (normalised to a)
    :param aspect_ratio: Aspect ratio of the plasma
    :param toroidal_mode_number: Toroidal mode number
    :param magnetic_shear: Magnetic shear at r_s

    :return: Diffusion length scale normalised to plasma minor radius a.
    """
    return (64*chi_perp/chi_parallel)**(1/4) * \
        (aspect_ratio*r_s/(toroidal_mode_number*magnetic_shear))**(1/2)

def chi_perp_ratio(diff_width: float,
                   r_s: float,
                   aspect_ratio: float,
                   toroidal_mode_number: int,
                   magnetic_shear: float) -> float:
    """
    Calculate the inverse of diffusion_width to solve for 
    chi_perp/chi_parallel

    See :func:`tearing_mode_solver.outer_region_solver.diffusion_width`

    :param diff_width: Diffusion width (normalised to a)
    :param aspect_ratio: Aspect ratio of the plasma
    :param toroidal_mode_number: Toroidal mode number
    :param magnetic_shear: Magnetic shear

    :return The ratio chi_perp/chi_parallel
    """
    return (diff_width**4/64)/(aspect_ratio*r_s/(toroidal_mode_number*magnetic_shear))**2

def curvature_stabilisation(diff_width: float,
                            resistive_interchange: float) -> float:
    """
    Calculate the curvature stabilisation modification to Delta'
    (Lutjens 2001, eq 3)

    The convention here is to normalise to minor radius for consistency
    with delta_prime() above. I.e. a*Delta'

    :param diff_width: Diffusion width (normalised to minor radius)
    :param resistive_interchange: Resistive interchange parameter D_R

    :return: Curvature stabilisation term normalised to minor radius
    """
    return np.sqrt(2)*np.pi**1.5 * resistive_interchange/diff_width


def curvature_stabilisation_non_linear(diff_width: float,
                                       resistive_interchange: float,
                                       magnetic_island_width: float) -> float:
    """
    Calculate the non-linear curvature stabilisation modification to Delta'
    (Lutjens 2001, eq 4)

    The convention here is to normalise to minor radius for consistency
    with delta_prime() above. I.e. a*Delta'

    :param diff_width: Diffusion width (normalised to minor radius)
    :param resistive_interchange: Resistive interchange parameter D_R
    :param magnetic_island_width: Magnetic island width (normalised to minor radius)

    :return: Curvature stabilisation term normalised to minor radius
    """
    return 6.35 * (
        resistive_interchange/
        (0.65*diff_width**2 + magnetic_island_width**2)**0.5
    )



def layer_width(poloidal_mode: int,
                toroidal_mode: int,
                lundquist_number: float,
                q_profile: List[Tuple[float, float]]) -> float:
    """
    Calculate the thickness of the resistive layer in the linear regime.
    """
    m = poloidal_mode
    n = toroidal_mode
    S = lundquist_number

    try:
        tm = solve_system(m, n, q_profile)
    except ValueError:
        return np.nan

    delta_p, gr = growth_rate(m, n, S, q_profile)

    s = magnetic_shear(q_profile, tm.r_s)

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

@np.vectorize
def island_width(psi_rs: float,
                 r_s: float,
                 poloidal_mode: int,
                 toroidal_mode: int,
                 magnetic_shear: float) -> float:
    """
    Helical width of a magnetic island (maximum distance between separatrices
    of the magnetic island).
    """
    pre_factor = poloidal_mode*psi_rs/(toroidal_mode*magnetic_shear)
    if pre_factor >= 0.0:
        return 4.0*np.sqrt(pre_factor)

    return 0.0

def delta_prime_nl_yu(tm: OuterRegionSolution,
                         island_width: float) -> float:
    """
    Non-linear discontinuity parameter calculation using the solution of the
    perturbed flux.

    This uses the Delta' scheme derived in Yu2004, i.e.

    $Delta'_{CL}(w) = Delta'(0) - a_{nl} w$,

    where a_{nl} is a coefficient which is either calculated or
    assigned as a free parameter to be fit.

    Parameters
    ----------
    tm : OuterRegionSolution
        Solution to the reduced MHD equation obtained from solve_system.
    island_width : float
        Width of the magnetic island.

    Returns
    -------
    float
        Delta' value.
    """
    a_nl = 44.024 # Placeholder

    r_min = tm.r_s
    dpsi_dr_min = tm.dpsi_dr_f_func(r_min)

    r_max = tm.r_s
    dpsi_dr_max = tm.dpsi_dr_b_func(r_max)

    psi_plus = tm.psi_backwards[-1]

    delta_p = (dpsi_dr_max - dpsi_dr_min)/psi_plus - a_nl * island_width

    return delta_p


def delta_prime_non_linear(tm: OuterRegionSolution,
                           island_width: float,
                           epsilon: float = 1e-10) -> float:
    """
    Non-linear discontinuity parameter calculation using the solution of the
    perturbed flux.

    Parameters
    ----------
    tm : OuterRegionSolution
        Solution to the reduced MHD equation obtained from solve_system.
    island_width : float
        Width of the magnetic island.
    epsilon : float, optional
        The tolerance needed for the lower and upper tearing mode solutions to
        match. The default is 1e-10.

    Raises
    ------
    ValueError
        Raised if upper and lower TM solutions don't match.

    Returns
    -------
    float
        Linear delta' value.

    """
    psi_plus = tm.psi_backwards[-1]
    psi_minus = tm.psi_forwards[-1]

    if abs(psi_plus - psi_minus) > epsilon:
        raise ValueError(
            f"""Forwards and backward solutions
            should be equal at resonant surface.
            (psi_plus={psi_plus}, psi_minus={psi_minus})."""
        )

    # return delta_prime_nl_yu(tm, island_width)

    r_min = tm.r_s - island_width/2.0
    dpsi_dr_min = tm.dpsi_dr_f_func(r_min)

    r_max = tm.r_s + island_width/2.0
    dpsi_dr_max = tm.dpsi_dr_b_func(r_max)

    delta_p = (dpsi_dr_max - dpsi_dr_min)/psi_plus

    return delta_p

def normalised_energy_integral(tm: OuterRegionSolution,
                               params: TearingModeParameters):
    m = params.poloidal_mode_number

    psi_rs = tm.psi_forwards[-1]

    r_f = tm.r_range_fwd[tm.r_range_fwd>0.0]
    psi_f = tm.psi_forwards[tm.r_range_fwd>0.0]/psi_rs
    dpsi_dr_f = tm.dpsi_dr_forwards[tm.r_range_fwd>0.0]/psi_rs

    integrand_f = m**2/r_f * psi_f**2 + r_f * dpsi_dr_f**2

    integral_f = simpson(integrand_f, r_f)

    # Backward part
    psi_b = tm.psi_backwards/psi_rs
    dpsi_dr_b = tm.dpsi_dr_backwards/psi_rs
    r_b = tm.r_range_bkwd

    integrand_b = m**2/r_b * psi_b**2 + r_b * dpsi_dr_b**2

    integral_b = simpson(integrand_b, r_b)

    # Abs in case either of the integrals are negative (they shouldn't be)
    return np.abs(integral_f) + np.abs(integral_b)

def energy(psi_rs: float, params: TearingModeParameters, norm_integral: float):
    """
    Calculate magnetic energy of the tearing mode.

    Note: norm_integral is calculated using the normalised_energy_integral()
    function above.

    psi_rs Can either be a float or an np.array.
    """
    return 2.0*np.pi**2 * params.R0 * (psi_rs**2) * norm_integral


def eta_to_lundquist_number(a: float, 
                            R_0: float,
                            B_tor: float, 
                            eta: float) -> float:
    """
    Calculate the Lundquist number from JOREK resistivity.

    See lab book eq:new-lundquist for derivation.

    :param a: Minor radius of plasma (metres)
    :param R_0: Major radius of plasma (metres)
    :param B_tor: Toroidal magnetic field (Tesla)
    :param eta: Resistivity at rational surface (r_s),
        [JOREK units]

    :return Lundquist number (unitless)
    """
    return a**2 * B_tor/(eta*R_0)


def alfven_frequency(R_0: float,
                     B_tor: float,
                     rho0: float) -> float:
    """
    Calculate the alfven frequency from plasma parameters

    :param R_0: Major radius of plasma (metres)
    :param B_tor: Toroidal magnetic field (Tesla)
    :param rho0: Central mass density of plasma (kg/m^3)
    """

    # Vacuum permeability
    mu0 = 4.0*np.pi*1e-7

    return B_tor/(R_0*(mu0*rho0)**0.5)
