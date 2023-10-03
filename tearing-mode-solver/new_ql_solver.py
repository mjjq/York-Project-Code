import numpy as np
from scipy.integrate import odeint, ode
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Tuple

from linear_solver import (
    OuterRegionSolution,
    gamma_constant,
    growth_rate,
    solve_system,
    scale_tm_solution,
    magnetic_shear,
    layer_width
)

from non_linear_solver import (
    delta_prime_non_linear,
    island_width
)

from helpers import savefig, savecsv, TimeDependentSolution

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

    linear_term = d2psi_dt2/dpsi_dt

    pre_factor = (non_linear_term + linear_term)

    # Avoid complex results by checking if argument is greater than zero
    # before fourth-rooting it.
    if pre_factor >= 0.0:
        return (pre_factor)**(1/4)/denominator

    return 0.0


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
                         mag_shear: float):
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
    delta_prime = delta_prime_non_linear(
        tm, w
    )

    

    s = mag_shear
    S = lundquist_number

    gamma = gamma_constant()

    linear_term = S*(n*s)**2 * (
        delta_prime * tm.r_s * psi/(gamma*S*dpsi_dt)
    )**4

    non_linear_term = -nu(psi, m, S, tm.r_s)

    d2psi_dt2 = dpsi_dt * (linear_term + non_linear_term)


    return [dpsi_dt, d2psi_dt2]



def solve_time_dependent_system(poloidal_mode: int,
                                toroidal_mode: int,
                                lundquist_number: float,
                                axis_q: float,
                                initial_scale_factor: float = 1.0,
                                t_range: np.array = np.linspace(0.0, 1e5, 10))\
                                    -> TimeDependentSolution:
    """
    Numerically integrate the quasi-linear flux time derivative of a tearing
    mode.

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
    #tm_s = scale_tm_solution(tm, initial_scale_factor)

    psi_t0 = initial_scale_factor#tm.psi_forwards[-1]

    # Calculate the initial growth rate of the mode using the linear theory
    # result. We hence assume that we are solving this system with an initially
    # small perturbed flux. If the flux were large, then this growth rate
    # would not be valid and an alternative growth rate must be used.
    delta_prime, linear_growth_rate = growth_rate(
        poloidal_mode,
        toroidal_mode,
        lundquist_number,
        axis_q
    )
    dpsi_dt_t0 = linear_growth_rate * psi_t0

    s = magnetic_shear(tm.r_s, poloidal_mode, toroidal_mode)

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
        s
    )

    # Set up return parameters.
    w_t = [init_island_width]
    psi = [psi_t0]
    dpsi_dt = [dpsi_dt_t0]
    d2psi_dt2 = [0.0]
    delta_primes = [delta_prime]
    times = [0.0]

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
            s
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
        
        delta_prime = delta_prime_non_linear(tm, init_island_width)

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



def ql_tm_vs_time():
    """
    Numerically solve the quasi-linear time-dependent tearing mode problem
    and plot the perturbed flux and layer width as functions of time.
    """
    m=2
    n=1
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e9, 1000000)

    ql_solution = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )
    times = ql_solution.times
    psi_t = ql_solution.psi_t
    dpsi_t = ql_solution.dpsi_dt
    w_t = ql_solution.w_t

    #plt.plot(w_t)
    #plt.show()

    #print("lgr: ", lin_growth_rate)
    #print("Threshold: ", ql_threshold)
    #print(psi_t)

    fig, ax = plt.subplots(1, figsize=(4,3.5))
    ax2 = ax.twinx()


    ax.plot(times, psi_t, label='Flux', color='black')

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \psi^{(1)}$)")

    ax2.plot(times, w_t, label='Normalised island width', color='red')
    ax2.set_ylabel(r"Normalised electrostatic modal width ($\hat{\delta}_{ql}$)")
    ax2.yaxis.label.set_color('red')

    ax.legend(prop={'size': 7}, loc='lower right')

    # ax.set_yscale('log')
    # ax2.set_yscale('log')
    ax.set_xscale('log')
    ax2.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_yscale('log')

    fig.tight_layout()
    #plt.show()

    fname = f"new_ql_tm_time_evo_(m,n,A,q0)=({m},{n},{solution_scale_factor},{axis_q})"
    savefig(fname)
    savecsv(fname, pd.DataFrame(asdict(ql_solution)))
    plt.show()

def ql_with_fit_plots():
    """
    Deprecated function.
    """
    m=4
    n=3
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e2, 100000)

    psi_t, w_t, tm0, dps, ql_threshold, s = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    lin_delta_prime, lin_growth_rate = growth_rate(
        m,
        n,
        lundquist_number,
        axis_q
    )

    #print("lgr: ", lin_growth_rate)
    #print("Threshold: ", ql_threshold)
    #print(psi_t)

    fig, ax = plt.subplots(1, figsize=(4,3))

    ql_time_min = time_from_flux(psi_t, times, 0.1*ql_threshold)
    ql_time_max = time_from_flux(psi_t, times, 10.0*ql_threshold)

    ax.fill_between(
        [ql_time_min, ql_time_max],
        2*[min(psi_t)],
        2*[max(psi_t)],
        alpha=0.3,
        label='Quasi-linear region'
    )

    ax.plot(times, psi_t, label='Flux', color='black')

    lin_times = times[np.where(times < ql_time_max)]
    ax.plot(
        lin_times,
        psi_t[0]*np.exp(lin_growth_rate*lin_times),
        label='Exponential fit'
    )

    nl_times = times[np.where(times >= ql_time_max)]
    psi_t0_nl = psi_t[np.abs(times-ql_time_max).argmin()]
    dp_nl = dps[np.abs(times-ql_time_max).argmin()]
    print(psi_t0_nl)
    ax.plot(
        nl_times,
        nl_parabola(
            tm0,
            s,
            lundquist_number,
            dp_nl,
            psi_t0_nl,
            nl_times
        ),
        label="Quadratic fit"
    )



    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")

    ax.legend(prop={'size': 7}, loc='lower right')

    # ax.set_yscale('log')
    # ax2.set_yscale('log')
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()
    #plt.show()
    savefig(
        f"ql_with_fitting_(m,n,A)=({m},{n},{solution_scale_factor})"
    )
    plt.show()

if __name__=='__main__':
    ql_tm_vs_time()
    #ql_with_fit_plots()
