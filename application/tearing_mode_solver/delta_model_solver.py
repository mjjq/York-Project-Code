import numpy as np
from scipy.integrate import ode
from tqdm import trange
from typing import Tuple

from tearing_mode_solver.outer_region_solver import (
    OuterRegionSolution,
    gamma_constant,
    growth_rate,
    solve_system,
    delta_prime_non_linear,
    island_width
)
from tearing_mode_solver.loizu_delta_prime import (
    calculate_coefficients, delta_prime_loizu, LoizuCoefficients
)

from tearing_mode_solver.profiles import (
    rational_surface_of_mode, magnetic_shear
)

from tearing_mode_solver.helpers import (
    TimeDependentSolution, TearingModeParameters
)

@np.vectorize
def nu(psi_rs: float,
       poloidal_mode: int,
       lundquist_number: float,
       r_s: float) -> float:
    """
    Calculate the \nu parameter to be used in the calculation for the
    quasi-linear layer width.

    Parameters:
        psi_rs: float
            The perturbed flux at the resonant surface
        poloidal_mode: int
            Poloidal mode of the tearing mode
        lundquist_number: float
            The Lundquist number
        r_s: float
            Location of the resonant surface (normalised to minor radius)
    """
    S = lundquist_number
    m = poloidal_mode

    return (0.5*m**2)*S*psi_rs**2/r_s**4

@np.vectorize
def mode_width(psi_rs: float,
                 dpsi_dt: float,
                 d2psi_dt2: float,
                 r_s: float,
                 poloidal_mode: int,
                 toroidal_mode: int,
                 mag_shear: float,
                 lundquist_number: float) -> float:
    """
    Calculate the quasi-linear mode width of the tearing mode.

    Parameters:
        psi_rs: float
            Perturbed flux at the resonant surface
        dpsi_dt: float
            First derivative in perturbed flux at the resonant surface
        d2psi_dt2: float
            Second derivative in perturbed flux at the resonant surface
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
    """
    S = lundquist_number
    n = toroidal_mode
    m = poloidal_mode
    s = mag_shear

    denominator = S**(1/4)*(n*s)**(1/2)

    non_linear_term = nu(psi_rs, m, S, r_s)

    ps_corr = np.sqrt(1+2*(m/n)**2)

    linear_term = ps_corr * d2psi_dt2/dpsi_dt

    pre_factor = (non_linear_term + linear_term)

    # Avoid complex results by checking if argument is greater than zero
    # before fourth-rooting it.
    if pre_factor >= 0.0:
        return (pre_factor)**(1/4)/denominator

    return 0.0

def mode_width_precalc(params: TearingModeParameters,
                       data: TimeDependentSolution) -> float:
    r_s = rational_surface_of_mode(
        params.q_profile,
        params.poloidal_mode_number,
        params.toroidal_mode_number
    )

    mag_shear = magnetic_shear(params.q_profile, r_s)

    return mode_width(
        data.psi_t,
        data.dpsi_dt,
        data.d2psi_dt2,
        r_s,
        params.poloidal_mode_number,
        params.toroidal_mode_number,
        mag_shear,
        params.lundquist_number
    )


def quasi_linear_threshold(toroidal_mode: int,
                           r_s: float,
                           mag_shear: float,
                           lundquist_number: float,
                           delta_prime_linear: float) -> float:
    """
    Calculate the threshold perturbed flux needed for a tearing mode to be
    considered to be in the quasi-linear regime.

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


#ode_island_width = []

def flux_time_derivative(time: float,
                         var: Tuple[float, float],
                         tm: OuterRegionSolution,
                         poloidal_mode: int,
                         toroidal_mode: int,
                         lundquist_number: float,
                         mag_shear: float,
                         loizu_coefs: LoizuCoefficients):
    """
    Calculate first and second order time derivatives of the perturbed flux
    in the inner region of the tearing mode using the quasi-linear time
    evolution equations (delta model).

    This is passed to scipy's ODE function to be integrated.

    Parameters:
        time: float
            The current time of the simulation.
        var: Tuple[float, float]
            Tuple containing the perturbed flux and rate of change of perturbed
            flux at the current time
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
        init_island_width: float
            Quasi-linear island width at the current time.
        delta_prime: float
            The non-linear discontinuity parameter at the current time.
    """

    psi, dpsi_dt = var
    m = poloidal_mode
    n = toroidal_mode

    w = island_width(psi, tm.r_s, m, n, mag_shear)
    delta_prime = delta_prime_loizu(w, loizu_coefs)

    

    s = mag_shear
    S = lundquist_number

    gamma = gamma_constant()

    q_rs = m/n
    # Remove pfirsch-schluter inertial correction
    # since we are not simulating a high beta
    # plasma, see Brunetti MHD report.
    ps_corr = 1.0#(1+2*q_rs**2)**(1/2)

    if np.abs(dpsi_dt/psi) > 1e-20:
        linear_term = S*(n*s)**2 * (
        delta_prime * tm.r_s * psi/(gamma*S*dpsi_dt)
        )**4
    else:
        linear_term = 0.0

    non_linear_term = -nu(psi, m, S, tm.r_s)

    d2psi_dt2 = dpsi_dt * (linear_term + non_linear_term) / ps_corr


    return [dpsi_dt, d2psi_dt2]



def solve_time_dependent_system(params: TearingModeParameters,
                                t_range: np.array = np.linspace(0.0, 1e5, 10))\
                                    -> TimeDependentSolution:
    """
    Numerically integrate the quasi-linear flux time derivative of a tearing
    mode.

    Parameters:
        params: TearingModeParameters
            Parameters used in the tearing mode.
        t_range: np.array
            Array of time values to record. Each element will have an associated
            perturbed flux, derivative etc calculated for that time.
    """
    poloidal_mode = params.poloidal_mode_number
    toroidal_mode = params.toroidal_mode_number
    lundquist_number = params.lundquist_number

    tm = solve_system(params)
    loizu_coefs = calculate_coefficients(params)
    
    #tm_s = scale_tm_solution(tm, initial_scale_factor)

    psi_t0 = params.initial_flux#tm.psi_forwards[-1]

    # Calculate the initial growth rate of the mode using the linear theory
    # result. We hence assume that we are solving this system with an initially
    # small perturbed flux. If the flux were large, then this growth rate
    # would not be valid and an alternative growth rate must be used.
    delta_prime, linear_growth_rate = growth_rate(
        poloidal_mode,
        toroidal_mode,
        lundquist_number,
        params.q_profile,
        tm
    )
    print(delta_prime, linear_growth_rate)
    dpsi_dt_t0 = linear_growth_rate * psi_t0

    s = magnetic_shear(params.q_profile, tm.r_s)

    # Calculate the initial width of the magnetic island using the linear layer
    # width.
    init_island_width = island_width(
        psi_t0,
        tm.r_s,
        poloidal_mode,
        toroidal_mode,
        s
    )



    t0 = t_range[0]
    tf = t_range[-1]
    dt = t_range[1]-t_range[0]
    # Use the vode algorithm provided by ODE. This is more stable than lsoda.
    r = ode(flux_time_derivative).set_integrator(
        'lsoda'#'dop853'#, atol=1, rtol=1, max_step=dt, first_step=dt
    )
    r.set_initial_value((psi_t0, dpsi_dt_t0), t0)
    r.set_f_params(
        tm,
        poloidal_mode,
        toroidal_mode,
        lundquist_number,
        s,
        loizu_coefs
    )

    # Set up return parameters.
    w_t = [init_island_width]
    psi = [psi_t0]
    dpsi_dt = [dpsi_dt_t0]
    d2psi_dt2 = [0.0]
    delta_primes = [delta_prime]
    times = [t0]

    # Loop through time range and integrate after each time. The island width
    # is updated outside of the derivative function as it relies on the
    # first and second time derivatives from the previous iteration.
    tqdm_range = trange(len(t_range)-1, desc='Time: ', leave=True)
    for i in tqdm_range:
        if not r.successful():
            #print("Unsuccessful. Breaking")
            #break
            pass
        tqdm_range.set_description(f"Time: {r.t:.2f}", refresh=True)
        
        #print(r.t, dt)

        # Integrate then calculate derivatives.
        psi_now, dpsi_dt_now = r.integrate(r.t+dt)

        _, d2psi_dt2_now = flux_time_derivative(
            r.t,
            (psi_now, dpsi_dt_now),
            tm,
            poloidal_mode,
            toroidal_mode,
            lundquist_number,
            s,
            loizu_coefs
            #init_island_width,
            #delta_prime
        )

        times.append(r.t)
        psi.append(psi_now)
        dpsi_dt.append(dpsi_dt_now)
        w_t.append(init_island_width)
        d2psi_dt2.append(d2psi_dt2_now)

        # Use the derivatives to calculate new island width.
        init_island_width = island_width(
            psi_now,
            tm.r_s,
            poloidal_mode,
            toroidal_mode,
            s
        )
        
        delta_prime = delta_prime_loizu(init_island_width, loizu_coefs)

        delta_primes.append(delta_prime)

        #r.set_f_params(
            #tm,
            #poloidal_mode,
            #toroidal_mode,
            #lundquist_number,
            #s,
            #init_island_width,
            #delta_prime
        #)

    return TimeDependentSolution(
        np.squeeze(times),
        np.squeeze(psi),
        np.squeeze(dpsi_dt),
        np.squeeze(d2psi_dt2),
        np.squeeze(w_t),
        np.squeeze(delta_primes)
    )

def time_from_flux(psi: np.array,
                   times: np.array,
                   target_psi: float):
    """
    Given an array of perturbed fluxes and times, find the closest time
    associated with the target flux (target_psi).
    """
    min_index = np.abs(psi - target_psi).argmin()
    return times[min_index]

