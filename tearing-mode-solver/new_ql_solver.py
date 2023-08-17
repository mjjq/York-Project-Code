import numpy as np
from scipy.integrate import odeint, ode
from matplotlib import pyplot as plt
from tqdm import tqdm

from linear_solver import (
    TearingModeSolution,
    gamma_constant,
    growth_rate,
    solve_system,
    scale_tm_solution,
    magnetic_shear,
    layer_width
)

from non_linear_solver import (
    delta_prime_non_linear
)

from pyplot_helper import savefig

@np.vectorize
def island_width(psi_rs: float,
                 dpsi_dt: float,
                 d2psi_dt2: float,
                 r_s: float,
                 poloidal_mode: float,
                 toroidal_mode: float,
                 mag_shear: float,
                 lundquist_number: float) -> float:
    S = lundquist_number
    n = toroidal_mode
    m = poloidal_mode
    s = mag_shear

    denominator = S*(n*s)**2

    non_linear_term = (0.5*m**2)*S*psi_rs**2/r_s**4

    linear_term = d2psi_dt2/dpsi_dt

    pre_factor = non_linear_term + linear_term

    if pre_factor >= 0.0:
        return (pre_factor/denominator)**(1/4)

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


#ode_island_width = []

def flux_time_derivative(time: float,
                         var: float,
                         tm: TearingModeSolution,
                         poloidal_mode: int,
                         toroidal_mode: int,
                         lundquist_number: float,
                         mag_shear: float,
                         init_island_width: float,
                         epsilon: float = 1e-5):

    psi, dpsi_dt = var

    #global ode_island_width
    w = init_island_width

    m = poloidal_mode
    n = toroidal_mode

    s = mag_shear
    S = lundquist_number

    delta_prime = delta_prime_non_linear(tm, w)
    gamma = gamma_constant()

    linear_term = S*(n*s)**2 * (
        delta_prime * tm.r_s * psi/(gamma*S*dpsi_dt)
    )**4

    non_linear_term = -(0.5*m**2)*S*(psi**2)/tm.r_s**4

    d2psi_dt2 = dpsi_dt * (linear_term + non_linear_term)

    #w_new = island_width(
        #psi,
        #dpsi_dt,
        #d2psi_dt2,
        #tm.r_s,
        #m,
        #n,
        #s,
        #S
    #)

    #print(w, dpsi_dt, d2psi_dt2)

    #ode_island_width.append(w_new)
    
    #print(f"iter time={time}")

    return [dpsi_dt, d2psi_dt2]



def solve_time_dependent_system(poloidal_mode: int,
                                toroidal_mode: int,
                                lundquist_number: float,
                                axis_q: float,
                                initial_scale_factor: float = 1.0,
                                t_range: np.array = np.linspace(0.0, 1e5, 10)):

    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)
    #tm_s = scale_tm_solution(tm, initial_scale_factor)

    psi_t0 = initial_scale_factor#tm.psi_forwards[-1]
    lin_delta_prime, linear_growth_rate = growth_rate(
        poloidal_mode,
        toroidal_mode,
        lundquist_number,
        axis_q
    )
    dpsi_dt_t0 = linear_growth_rate * psi_t0

    init_island_width = layer_width(
        poloidal_mode,
        toroidal_mode,
        lundquist_number,
        axis_q
    )

    s = magnetic_shear(tm.r_s, poloidal_mode, toroidal_mode)


    t0 = t_range[0]
    tf = t_range[-1]
    dt = t_range[1]-t_range[0]
    r = ode(flux_time_derivative).set_integrator('zvode', method='bdf')
    r.set_initial_value((psi_t0, dpsi_dt_t0), t0)
    r.set_f_params(
        tm,
        poloidal_mode,
        toroidal_mode,
        lundquist_number,
        s,
        init_island_width
    )

    w_t = [init_island_width]
    psi = [psi_t0]
    dpsi_dt = [dpsi_dt_t0]

    for i in tqdm(range(len(t_range)-1)):
        if not r.successful():
            print("Unsuccessful. Breaking")
            break

        psi_now, dpsi_dt_now = r.integrate(r.t+dt)

        _, d2psi_dt2_now = flux_time_derivative(
            r.t,
            (psi_now, dpsi_dt_now),
            tm,
            poloidal_mode,
            toroidal_mode,
            lundquist_number,
            s,
            init_island_width
        )

        psi.append(psi_now)
        dpsi_dt.append(dpsi_dt_now)
        w_t.append(init_island_width)

        init_island_width = island_width(
            psi_now,
            dpsi_dt_now,
            d2psi_dt2_now,
            tm.r_s,
            poloidal_mode,
            toroidal_mode,
            s,
            lundquist_number
        )

        r.set_f_params(
            tm,
            poloidal_mode,
            toroidal_mode,
            lundquist_number,
            s,
            init_island_width
        )

    #result = odeint(
        #flux_time_derivative,
        #(psi_t0, dpsi_dt_t0),
        #t_range,
        #args = (
            #tm,
            #poloidal_mode,
            #toroidal_mode,
            #lundquist_number,
            #s
        #)
    #)


    # We get weird numerical bugs sometimes returning large or nan values.
    # Set these to zero.
    #psi_t[np.abs(psi_t) > 1e10] = 0.0
    #psi_t[np.argwhere(np.isnan(psi_t))] = 0.0

    #dps = [delta_prime_non_linear(tm, w) for w in w_t]

    return (
        np.squeeze(psi),
        np.squeeze(dpsi_dt),
        np.squeeze(w_t)
    )

def time_from_flux(psi: np.array,
                   times: np.array,
                   target_psi: float):
    min_index = np.abs(psi - target_psi).argmin()
    return times[min_index]



def ql_tm_vs_time():
    m=2
    n=1
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e3, 100)

    psi_t, dpsi_t, w_t = solve_time_dependent_system(
        m, n, lundquist_number, axis_q, solution_scale_factor, times
    )

    plt.plot(w_t)
    #plt.show()

    #print("lgr: ", lin_growth_rate)
    #print("Threshold: ", ql_threshold)
    #print(psi_t)

    fig, ax = plt.subplots(1, figsize=(4,3))
    ax2 = ax.twinx()


    ax.plot(times, psi_t, label='Flux', color='black')

    ax.set_xlabel(r"Normalised time ($\bar{\omega}_A t$)")
    ax.set_ylabel(r"Normalised perturbed flux ($\delta \hat{\psi}^{(1)}$)")

    #ax2.plot(times, w_t, label='Normalised island width', color='red')
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
    savefig(
        f"new_ql_tm_time_evo_(m,n,A)=({m},{n},{solution_scale_factor})"
    )
    plt.show()

def ql_with_fit_plots():
    m=4
    n=3
    lundquist_number = 1e8
    axis_q = 1.0
    solution_scale_factor = 1e-10

    times = np.linspace(0.0, 1e8, 10000)

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
