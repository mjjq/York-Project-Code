import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

from tearing_mode_solver.outer_region_solver import (
    OuterRegionSolution,
    gamma_constant,
    growth_rate,
    solve_system,
    scale_tm_solution,
    magnetic_shear,
    delta_prime_non_linear,
    island_width
)

from tearing_mode_solver.helpers import (
    savefig, TimeDependentSolution, dataclass_to_disk
)
from tearing_mode_solver.algebraic_fitting import nl_parabola

@np.vectorize
def modal_width(psi_rs: float,
                 r_s: float,
                 poloidal_mode: float,
                 toroidal_mode: float,
                 mag_shear: float,
                 lundquist_number: float,
                 linear_growth_rate: float) -> float:
    """
    Calculate the quasi-linear electrostatic modal width of the tearing mode
    (gamma model).

    Parameters:
        psi_rs: float
            Perturbed flux at the resonant surface
        r_s: float
            Location of the resonant surface normalised to plasma minor radius
        poloidal_mode: int
            Poloidal mode number of the tearing mode
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        mag_shear: float
            Magnetic shear at the resonant surface. See magnetic_shear() in
            linear_solver.py
        lundquist_number: float
            The Lundquist number
        linear_growth_rate: float
            The growth rate of the tearing mode in the linear regime. See
            growth_rate() in linear_solver.py
    """

    denominator = (lundquist_number**(1/4))*(
        toroidal_mode*mag_shear)**(1/2)

    pre_factor = linear_growth_rate + \
        0.5*lundquist_number*(poloidal_mode*psi_rs)**2 / r_s**4

    if pre_factor >= 0.0:
        return ((pre_factor)**(1/4))/denominator

    return 0.0


def quasi_linear_threshold(toroidal_mode: int,
                           r_s: float,
                           mag_shear: float,
                           lundquist_number: float,
                           delta_prime_linear: float):
    """
    Calculate the threshold perturbed flux needed for a tearing mode to be
    considered to be in the quasi-linear regime (from the gamma model).

    Parameters:
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        r_s: float
            Location of the resonant surface normalised to the plasma minor
            radius
        mag_shear: float
            Magnetic shear at the resonant surface. See magnetic_shear() in
            linear_solver.py
        lundquist_number: float
            The Lundquist number
        delta_prime_linear: float
            Discontinuity parameter calculated in the linear regime. See
            delta_prime() in linear_solver.py
    """

    g = gamma_constant()
    n = toroidal_mode
    s = mag_shear
    S = lundquist_number
    return np.sqrt(2)*(g**(-2/5))*((n*s)**(1/5))* \
        (r_s**(12/5))*(S**(-4/5))*(delta_prime_linear**(2/5))

def flux_time_derivative(psi: float,
                         time: float,
                         tm: OuterRegionSolution,
                         poloidal_mode: int,
                         toroidal_mode: int,
                         lundquist_number: float,
                         mag_shear: float,
                         linear_growth_rate: float,
                         epsilon: float = 1e-5):
    """
    Calculate first order time derivative of the perturbed flux
    in the inner region of the tearing mode using the quasi-linear time
    evolution equations (gamma model).

    This is passed to scipy's ODE function to be integrated.

    Parameters:
        psi: float
            Perturbed flux at the resonant surface and current time
        time: float
            The current time of the simulation.
        tm: OuterRegionSolution
            The outer solution of the current tearing mode
        poloidal_mode: int
            Poloidal mode number of the tearing mode
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        lundquist_number: float
            The Lundquist number
        mag_shear: float
            Magnetic shear at the resonant surface. See magnetic_shear() in
            linear_solver.py
        delta_prime: float
            The non-linear discontinuity parameter at the current time.
    """

    m = poloidal_mode
    n = toroidal_mode

    #if psi[0]<0.0:
    #    print(f"Warning, negative flux at {time}. Setting to zero.")
    #    psi[0]=0.0
    # if psi[0]>1e-5:
    #     print("Warning, weirdly high psi value")
    # if np.isnan(psi[0]):
    #     print("Warning, psi is nan")

    s = mag_shear
    w = island_width(
        psi, tm.r_s, m, n, s
    )
    delql = modal_width(
        psi, tm.r_s, m, n, s, lundquist_number, linear_growth_rate
    )

    delta_prime = delta_prime_non_linear(tm, w)

    gamma = gamma_constant()

    dpsi_dt = tm.r_s * psi * delta_prime / (gamma*delql*lundquist_number)
    
    # print(psi)
    # print(w)
    #print(dpsi_dt/psi)
    # print()

    return dpsi_dt


def solve_time_dependent_system(poloidal_mode: int,
                                toroidal_mode: int,
                                lundquist_number: float,
                                axis_q: float,
                                initial_scale_factor: float = 1.0,
                                t_range: np.array = np.linspace(0.0, 1e5, 10)):
    """
    Numerically integrate the quasi-linear flux time derivative of a tearing
    mode (gamma model).

    Parameters:
        poloidal_mode: int
                Poloidal mode number of the tearing mode
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        lundquist_number: float
            The Lundquist number
        axis_q: float
            The on-axis equilibrium safety factor
        initial_scale_factor: float
            The value of the perturbed flux at the resonant surface at t=0
        t_range: np.array
            Array of time values to record. Each element will have an associated
            perturbed flux, derivative etc calculated for that time.
    """

    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)

    psi_t0 = initial_scale_factor

    s = magnetic_shear(tm.r_s, poloidal_mode, toroidal_mode)

    lin_delta_prime, lin_growth_rate = growth_rate(
        poloidal_mode,
        toroidal_mode,
        lundquist_number,
        axis_q
    )
    print(lin_growth_rate)

    psi_t = odeint(
        flux_time_derivative,
        psi_t0,
        t_range,
        args = (
            tm,
            poloidal_mode,
            toroidal_mode,
            lundquist_number,
            s,
            lin_growth_rate
        )
    )

    # We get weird numerical bugs sometimes returning large or nan values.
    # Set these to zero.
    psi_t[np.abs(psi_t) > 1e10] = 0.0
    psi_t[np.argwhere(np.isnan(psi_t))] = 0.0

    w_t = np.squeeze(
        modal_width(
            psi_t, tm.r_s,
            poloidal_mode, toroidal_mode,
            s, lundquist_number,
            lin_growth_rate
        )
    )

    dps = [delta_prime_non_linear(tm, w) for w in w_t]
    
    ql_threshold = quasi_linear_threshold(
        toroidal_mode,
        tm.r_s,
        s,
        lundquist_number,
        lin_delta_prime
    )

    # Use spline to construct the flux as a continuous function, then use this
    # to get first and second-order derivatives.
    psi_spline = UnivariateSpline(t_range, psi_t, s=0)
    dpsi_dt = psi_spline.derivative()(t_range)
    d2psi_dt2 = psi_spline.derivative().derivative()(t_range)

    sol = TimeDependentSolution(
        t_range,
        np.squeeze(psi_t),
        np.squeeze(dpsi_dt),
        np.squeeze(d2psi_dt2),
        np.squeeze(w_t),
        np.array(dps)
    )

    return sol, tm, ql_threshold, s

def time_from_flux(psi: np.array,
                   times: np.array,
                   target_psi: float):
    """
    Given an array of perturbed fluxes and times, find the closest time
    associated with the target flux (target_psi).
    """
    min_index = np.abs(psi - target_psi).argmin()
    return times[min_index]


