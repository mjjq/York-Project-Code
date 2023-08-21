import numpy as np
from dataclasses import dataclass, fields
import pandas as pd
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from scipy.integrate import quad, simpson
from tqdm import tqdm, trange
from typing import Tuple
import os

from y_sol import Y
from new_ql_solver import nu, island_width
from helpers import savefig, classFromArgs, TimeDependentSolution
from linear_solver import magnetic_shear, rational_surface

def del_ql_full(sol: TimeDependentSolution,
                poloidal_mode: int,
                toroidal_mode: int,
                lundquist_number: float,
                mag_shear: float,
                r_s: float,
                x_range: Tuple[float, float],
                dx: float = 0.01):

    times = sol.times
    w_t_func = UnivariateSpline(times, sol.w_t, s=0)
    dw_dt_func = w_t_func.derivative()

    xmin, xmax = x_range
    xs = np.arange(xmin, xmax, dx)
    ys = Y(xs)
    ys_func = UnivariateSpline(xs, ys, s=0)
    d2ydx2 = ys_func.derivative().derivative()
    d3ydx3 = d2ydx2.derivative()

    del_dot_term = dw_dt_func(times)/w_t_func(times)
    psi_dot_term = sol.d2psi_dt2/sol.dpsi_dt
    nu_value = nu(sol.psi_t, poloidal_mode, lundquist_number, r_s)

    pre_factor = 1.0/( lundquist_number * (toroidal_mode*mag_shear)**2)

    deltas = []

    tqdm_range = trange(len(xs), leave=True)
    for i in tqdm_range:
        x = xs[i]
        M = x*d3ydx3(x)/d2ydx2(x)
        del_dot = - del_dot_term * (3.0+M)
        psi_dot = psi_dot_term

        delta_pow_4 = pre_factor*(nu_value + psi_dot + del_dot)

        delta_pow_4[delta_pow_4<0.0] = 0.0

        delta_at_x = delta_pow_4**(1/4)

        deltas.append(delta_at_x)

    deltas = np.array(deltas)

    return deltas, times, xs

def simple_integration():
    xs = np.linspace(-10.0, 10.0, 100)
    ys = Y(xs)
    
    int_result = simpson(
        (1.0+xs*ys), x=xs
    )
    
    print(int_result)
    

def delta_prime_full(delta_qls: np.ndarray,
                     xs: np.array,
                     times: np.array,
                     psi_t: np.array,
                     dpsi_dt: np.array,
                     w_t: np.array,
                     r_s: float,
                     lundquist_number: float):
    
    ys = Y(xs)
    
    
    delta_primes = []
    tqdm_range = trange(len(times), leave=True)
    for i in tqdm_range:
        t= times[i]
        delta_ql_x = delta_qls[:,i]
        delta_orig = w_t[i]
        
        int_result = simpson(
            (1.0+(delta_orig/delta_ql_x)*xs*ys), x=xs
        )

        psi = psi_t[i]
        dpsi = dpsi_dt[i]
        
        delta_primes.append(lundquist_number*dpsi*delta_orig*int_result/(psi*r_s))
    
    return np.array(delta_primes)

def convergence_of_delta_prime():
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "./output/18-08-2023_16:41_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

    delta_ql_orig = island_width(
        ql_sol.psi_t,
        ql_sol.dpsi_dt,
        ql_sol.d2psi_dt2,
        r_s,
        m,
        n,
        s,
        S
    )

    simple_integration()

    x_lims = [1.0, 1.5, 5.0]


    fig_dp, ax_dp = plt.subplots(1, figsize=(4,4))
    ax_dp.plot(
        ql_sol.times,
        ql_sol.delta_primes,
        label=r"Approximate $a\Delta'$",
        color='black'
    )
    ax_dp.set_xscale('log')

    for xlim in x_lims:
        delqls, times, xs = del_ql_full(
            ql_sol, m, n, S, s, r_s,
            (-xlim, xlim),
            0.1
        )
        #delqls = delqls[:,::1000]

        delta_primes = delta_prime_full(
            delqls,
            xs,
            times,
            ql_sol.psi_t,
            ql_sol.dpsi_dt,
            delta_ql_orig,
            r_s,
            S
        )

        times_f = times[times>1e3]
        delta_primes_f = delta_primes[-len(times_f):]
        print(times_f.shape)
        print(delta_primes_f.shape)
        ax_dp.plot(
            times_f, delta_primes_f, label=r"Exact $a\Delta'$"+f"(X={xlim})"
        )

    ax_dp.legend()
    ax_dp.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    ax_dp.set_ylabel(r"$a\Delta'$")
    ax_dp.set_xlim(left=1e3)
    #ax_dp.set_ylim(bottom=1.0)
    fig_dp.tight_layout()

    orig_fname, ext = os.path.splitext(os.path.basename(fname))
    savefig(f"{orig_fname}_delta_prime_convergence")


def constant_psi_approx():
    m=2
    n=1
    S=1e8
    axis_q=1.2
    r_s=rational_surface(m/n)
    s=magnetic_shear(r_s, m, n)

    fname = "./output/20-08-2023_17:12_new_ql_tm_time_evo_(m,n,A,q0)=(2,1,1e-10,1.2).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

    delta_ql_orig = island_width(
        ql_sol.psi_t,
        ql_sol.dpsi_dt,
        ql_sol.d2psi_dt2,
        r_s,
        m,
        n,
        s,
        S
    )

    simple_integration()

    x_lims = [20.0]


    fig_dp, ax_dp = plt.subplots(1, figsize=(4,4))

    #ax_dp.set_xscale('log')

    fig_time, ax_time = plt.subplots(1, figsize=(4,4))
    ax_time.set_xscale('log')

    for xlim in x_lims:
        delqls, times, xs = del_ql_full(
            ql_sol, m, n, S, s, r_s,
            (-xlim, xlim),
            0.1
        )
        #delqls = delqls[:,::1000]

        delta_primes = delta_prime_full(
            delqls,
            xs,
            times,
            ql_sol.psi_t,
            ql_sol.dpsi_dt,
            delta_ql_orig,
            r_s,
            S
        )

        widths = delqls[-1]
        d_delta_primes = widths*delta_primes

        times_f = times[times>1e3]
        widths_f = widths[-len(times_f):]
        d_delta_primes_f = d_delta_primes[-len(times_f):]
        #ax_dp.plot(
            #widths_f, d_delta_primes_f, label=r"Exact $a\Delta'$"+f"(X={xlim})",
            #color='black'
        #)
        ax_time.plot(
            times_f, d_delta_primes_f, label=r"Exact $a\Delta'$"+f"(X={xlim})",
            color='black'
        )

    #ax_dp.legend()
    ax_dp.set_xlabel(r"Layer width")
    ax_dp.set_ylabel(r"$\delta \Delta'$")
    ax_time.set_xlabel(r"Normalised time $\bar{\omega} t$")
    ax_time.set_ylabel(r"$\delta \Delta'$")
    #ax_dp.set_xlim(left=1e3)
    #ax_dp.set_ylim(bottom=1.0)
    fig_dp.tight_layout()
    fig_time.tight_layout()

    orig_fname, ext = os.path.splitext(os.path.basename(fname))
    savefig(f"{orig_fname}_const_psi_approx")

def convergence_of_growth_rate():
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "./output/18-08-2023_16:41_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

    delta_ql_orig = island_width(
        ql_sol.psi_t,
        ql_sol.dpsi_dt,
        ql_sol.d2psi_dt2,
        r_s,
        m,
        n,
        s,
        S
    )

    simple_integration()

    delqls, times, xs = del_ql_full(ql_sol, m, n, S, s, r_s)
    #delqls = delqls[:,::1000]

    delta_primes = delta_prime_full(
        delqls,
        xs,
        times,
        ql_sol.psi_t,
        ql_sol.dpsi_dt,
        r_s,
        S,
        (-10.0, 10.0),
        0.1
    )

    fig_dp, ax_dp = plt.subplots(1, figsize=(4,4))
    ax_dp.plot(times, delta_primes, label=r"Exact $\Delta'$")
    ax_dp.plot(times, ql_sol.delta_primes, label=r"Approximate $\Delta'$")

    # fig, ax = plt.subplots(1, figsize=(4,4))
    # plt.imshow(
    #     delqls,
    #     extent=[min(times), max(times), min(xs), max(xs)]
    # )

    # ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))

    fig2, ax2 = plt.subplots(1, figsize=(4,3))

    ax2.plot(times, delta_ql_orig, label=f'Recreated approximate soln')
    ax2.plot(times, ql_sol.w_t, label=f'Approximate solution')
    ax2.plot(times, delqls[-1,:], label=f'x={xs[-1]:.2f}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    for i in range(len(xs)//2, len(xs), 200):
        lineout = delqls[i,:]
        ax2.plot(times, lineout, label=f'x={xs[i]:.2f}')


    ax2.legend()

    fig_psi, ax_psi = plt.subplots(1)
    ax_psi.plot(times, ql_sol.psi_t)
    ax_w = ax_psi.twinx()

    ax_psi.set_xscale('log')
    ax_psi.set_yscale('log')

    ax_w.plot(times, ql_sol.w_t)
    ax_w.set_yscale('log')

if __name__=='__main__':
    constant_psi_approx()
    #convergence_of_delta_prime()

    plt.show()
