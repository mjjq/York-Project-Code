import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

from linear_solver import (
    TearingModeSolution,
    gamma_constant,
    growth_rate,
    solve_system,
    scale_tm_solution,
    magnetic_shear
)

from non_linear_solver import (
    delta_prime_non_linear,
    island_width
)

from helpers import savefig, TimeDependentSolution, dataclass_to_disk
from nl_td_solver import nl_parabola

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
                         tm: TearingModeSolution,
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
        tm: TearingModeSolution
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
        psi, tm.r_s, s
    )

    delta_prime = delta_prime_non_linear(tm, w)

    gamma = gamma_constant()

    dpsi_dt = tm.r_s * psi * delta_prime / (gamma*w*lundquist_number)
    
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

    times = np.linspace(0.0, 1e8, 100000)

    sol, tm0, ql_threshold, s = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )
    
    psi_t, w_t, dps = sol.psi_t, sol.w_t, sol.delta_primes

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
    ax2 = ax.twinx()

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

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")

    ax2.plot(times, w_t, label='Normalised island width', color='red')
    ax2.set_ylabel(r"Normalised layer width ($\hat{\delta}$)")
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
    fname = f"ql_tm_time_evo_(m,n,A)=({m},{n},{solution_scale_factor})"
    savefig(fname)
    dataclass_to_disk(fname, sol)
    plt.show()

def ql_with_fit_plots():
    """
    Plot quasi-linear solution to the perturbed flux in conjunction with an
    exponential fit for the linear regime and a quadratic fit in the strongly
    non-linear regime.
    """
    m=2
    n=1
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e8, 10000)

    sol, tm0, ql_threshold, s = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    psi_t, w_t, dps = sol.psi_t, sol.w_t, sol.delta_primes

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

from lmfit.models import ExponentialModel

def check_exponential_fit():
    """
    Solve quasi-linear tearing mode numerically and calculate RMS error in the
    exponential fit in the linear regime for increasing time intervals.
    """
    m=3
    n=2
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e8, 10000)

    sol, tm0, ql_threshold, s = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    psi_t, w_t, dps = sol.psi_t, sol.w_t, sol.delta_primes

    lin_delta_prime, lin_growth_rate = growth_rate(
        m,
        n,
        lundquist_number,
        axis_q
    )

    # Estimate time at which quasi-linear region begins. This is the time at
    # which the perturbed flux is 1/10th the amount of the threshold flux.
    ql_time_min = time_from_flux(psi_t, times, 0.1*ql_threshold)
    # Estimate time at which quasi-linear region ends and strongly non-linear
    # regime begins. This is 10 times the amount of the threshold flux.
    ql_time_max = time_from_flux(psi_t, times, 10.0*ql_threshold)

    # Define an array of upper values for the fitting interval.
    max_times = np.linspace(0.5*ql_time_min, 2.0*ql_time_max, 100)

    fig, ax = plt.subplots(1, figsize=(4,3))

    chisqrs = []

    for max_time in max_times:
        # Grab only the data in the interval 0 <= t <= max_time
        lin_filter = np.where(times < max_time)
        lin_times = times[lin_filter]
        lin_psi = psi_t[lin_filter]

        #linear_model = ExponentialModel()
        #params = linear_model.make_params(
        #    amplitude=psi_t[0],
        #    decay=-1.0/lin_growth_rate
        #)
        #result = linear_model.fit(lin_psi, params, x=lin_times)
        
        exp_fit = lin_psi[0]*np.exp(lin_growth_rate*lin_times)

        rms_frac_error = np.mean((1.0-lin_psi/exp_fit)**2)
        chisqrs.append(rms_frac_error)
        
        #chisqrs.append(result.chisqr)

    print(chisqrs)
    ax.plot(max_times, chisqrs, color='black')

    ax.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    ax.set_ylabel(r"Exponential fit RMS fractional error")

    #ax.set_xlim(left=0.0, right=2.0*ql_time_min)
    #ax.set_ylim(top=1.0)
    ax.set_yscale('log')
    ax.grid(which='major')

    ax.fill_between(
        [ql_time_min, ql_time_max],
        -200.0,
        200.0,
        alpha=0.3,
        label='Quasi-linear region'
    )

    ax.legend()
    
    ax.set_ylim(top=1.0)
    
    fig.tight_layout()
    
    savefig(
        f"frac_error_exp_fit_(m,n,A)=({m},{n},{solution_scale_factor})"
    )

    plt.show()

if __name__=='__main__':
    ql_tm_vs_time()
    #ql_with_fit_plots()
    #check_exponential_fit()
