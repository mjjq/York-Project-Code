import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

from linear_solver import (
    TearingModeSolution,
    gamma_constant,
    growth_rate,
    solve_system,
    scale_tm_solution,
    magnetic_shear
)

from non_linear_solver import (
    delta_prime_non_linear
)

from pyplot_helper import savefig

@np.vectorize
def island_width(psi_rs: float,
                 r_s: float,
                 poloidal_mode: float,
                 toroidal_mode: float,
                 magnetic_shear: float,
                 lundquist_number: float,
                 linear_growth_rate: float) -> float:
    denominator = (lundquist_number**(1/4))*(
        toroidal_mode*magnetic_shear)**(1/2)

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
        psi, tm.r_s, m, n, s, lundquist_number, linear_growth_rate
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

    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)
    #tm_s = scale_tm_solution(tm, initial_scale_factor)

    psi_t0 = initial_scale_factor#tm.psi_forwards[-1]

    s = magnetic_shear(tm.r_s, poloidal_mode, toroidal_mode)

    lin_delta_prime, lin_growth_rate = growth_rate(
        poloidal_mode,
        toroidal_mode,
        lundquist_number,
        axis_q
    )

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
        island_width(
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

    return np.squeeze(psi_t), w_t, tm, dps, ql_threshold, s

def time_from_flux(psi: np.array,
                   times: np.array,
                   target_psi: float):
    min_index = np.abs(psi - target_psi).argmin()
    return times[min_index]

def nl_parabola_coefficients(tm: TearingModeSolution,
                             mag_shear: float,
                             lundquist_number: float,
                             delta_prime_linear: float,
                             psi_0: float):
    c_0 = (tm.r_s**3 / (64*lundquist_number**2))\
        *mag_shear*delta_prime_linear**2
    c_1 = np.sqrt(psi_0) * (tm.r_s**3 * mag_shear)**0.5\
        * delta_prime_linear/(4*lundquist_number)
    c_2 = psi_0

    return c_0, c_1, c_2

def nl_parabola(tm: TearingModeSolution,
                mag_shear: float,
                lundquist_number: float,
                delta_prime_linear: float,
                psi_0: float,
                times: np.array):
    c_0, c_1, c_2 = nl_parabola_coefficients(
        tm,
        mag_shear,
        lundquist_number,
        delta_prime_linear,
        psi_0
    )

    new_times = times - times[0]

    return c_0*(new_times**2) + c_1*new_times + c_2

def ql_tm_vs_time():
    m=3
    n=2
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e8, 300)

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

    lin_times = times[np.where(times < ql_time_max)]
    ax.plot(
        lin_times,
        psi_t[0]*np.exp(lin_growth_rate*lin_times),
        label='Exponential time dependence (linear regime)'
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
        label="Quadratic time dependence (non-linear regime)"
    )



    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")

    ax2.plot(times, w_t, label='Normalised island width', color='red')
    ax2.set_ylabel(r"Normalised layer width ($\hat{\delta}$)")
    ax2.yaxis.label.set_color('red')
    
    ax.legend(prop={'size': 7})
    
    # ax.set_yscale('log')
    # ax2.set_yscale('log')
    ax.set_xscale('log')
    ax2.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_yscale('log')

    fig.tight_layout()
    #plt.show()
    savefig(
        f"ql_tm_time_evo_(m,n,A)=({m},{n},{solution_scale_factor})"
    )
    plt.show()

if __name__=='__main__':
    ql_tm_vs_time()
