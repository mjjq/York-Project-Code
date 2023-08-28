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
    """
    Calculate the full (unapproximated) quasi-linear layer width as a function
    of X and t.

    Parameters:
        sol: TimeDependentSolution
            The time-dependent tearing mode solution
        poloidal_mode: int
            Poloidal mode number of the tearing mode
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        lundquist_number: float
            The Lundquist number
        mag_shear: float
            Magnetic shear at the resonant surface
        r_s: float
            Location of the resonant surface normalised to the minor radius of
            the plasma.
        x_range: Tuple[float, float]
            Minimum and maximum bounds defining the range of X values to use in
            the calculation of delta(X,t)
        dx: float
            Distance between adjacent X-values in x_range

    Returns:
        deltas: np.ndarray
            2D array containing delta values as a function of X and t. First
            dimension corresponds to time, second dimension corresponds to
            space.
        times: np.array
            Array of times associated with the first dimension of deltas
        xs: np.array
            Array of X values associated with the second dimension of deltas

    """

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
    """
    Perform the approximate spatial integral for the inner layer solution to
    verify that we get the correct result. We should get ~2.12
    """
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
    """
    Calculate the full (unapproximated) discontinuity parameter of the
    quasi-linear inner layer solution. Integrates over the spatial component
    so that Delta' is a function of time only.

    Parameters:
        deltas: np.ndarray
            2D array containing delta values as a function of X and t. First
            dimension corresponds to time, second dimension corresponds to
            space.
        xs: np.array
            Array of X values associated with the second dimension of deltas
        times: np.array
            Array of times associated with the first dimension of deltas
        psi_t: np.array
            Perturbed flux at the resonant surface as a function of time
        dpsi_dt: np.array
            First time derivative in perturbed flux at resonant surface as a
            function of time.
        w_t: np.array
            Quasi-linear layer width as a function of time.
        r_s: float
            Location of the resonant surface normalised to the minor radius of
            the plasma.
        lundquist_number: float
            The Lundquist number.

    Returns:
        delta_primes: np.array
            Discontinuity parameter as a function of time.
    """
    
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
    """
    Demonstrate convergence of the unapproximated discontinuity parameter to the
    approximated value over a numerical solution to the quasi-linear equations.
    """
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
    """
    Test the constant-psi approximation using the full (unapproximated) value
    of the discontinuity parameter for a quasi-linear tearing mode solution.
    """
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
        ax_dp.plot(
            widths_f, d_delta_primes_f, label=r"Exact $a\Delta'$"+f"(X={xlim})",
            color='black'
        )
        #ax_time.plot(
        #    times_f, d_delta_primes_f, label=r"Exact $a\Delta'$"+f"(X={xlim})",
        #    color='black'
        #)

    ax_dp.set_xscale('log')
    #ax_dp.legend()
    ax_dp.set_xlabel(r"Layer width")
    ax_dp.set_ylabel(r"$\delta \Delta'$")
    ax_time.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    ax_time.set_ylabel(r"$\delta \Delta'$")
    #ax_dp.set_xlim(left=1e3)
    #ax_dp.set_ylim(bottom=1.0)
    fig_dp.tight_layout()
    fig_time.tight_layout()

    orig_fname, ext = os.path.splitext(os.path.basename(fname))
    savefig(f"{orig_fname}_const_psi_approx")

def convergence_of_growth_rate():
    """
    Demonstrate convergence of the unapproximated growth rate to the
    approximated value over a numerical solution to the quasi-linear equations.
    """
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

def plot_full_delql():
    """
    Plot the full quasi-linear layer width as a function of (X, t) on a heatmap.
    """
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "./output/18-08-2023_16:41_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

    x_range = (-4, 4)

    delqls, times, xs = del_ql_full(ql_sol, m, n, S, s, r_s, x_range)

    fig, ax = plt.subplots(1, figsize=(4.3,4))
    ax.set_xlim(left=1e4, right=3e5)
    im = plt.imshow(
        delqls,
        extent=[min(times), max(times), min(xs), max(xs)],
        vmax=0.002
    )
    ax.set_aspect(0.5*(max(times)-min(times))/(max(xs)-min(xs)))
    fig.colorbar(im, fraction=0.046, pad=0.04)

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax.set_xlabel(r"Normalised time $\bar{\omega}_A t$")
    ax.set_ylabel(r"X")
    fig.tight_layout()

    savefig(f"delta_heatmap_(m,n)=({m},{n})")


def plot_delql_terms(sol: TimeDependentSolution,
                poloidal_mode: int,
                toroidal_mode: int,
                lundquist_number: float,
                r_s: float,
                x_range: Tuple[float, float],
                plot_t: float,
                dx: float = 0.01):
    """
    Calculate the full (unapproximated) quasi-linear layer width as a function
    of X and t.

    Parameters:
        sol: TimeDependentSolution
            The time-dependent tearing mode solution
        poloidal_mode: int
            Poloidal mode number of the tearing mode
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        lundquist_number: float
            The Lundquist number
        mag_shear: float
            Magnetic shear at the resonant surface
        r_s: float
            Location of the resonant surface normalised to the minor radius of
            the plasma.
        x_range: Tuple[float, float]
            Minimum and maximum bounds defining the range of X values to use in
            the calculation of delta(X,t)
        dx: float
            Distance between adjacent X-values in x_range

    Returns:
        deltas: np.ndarray
            2D array containing delta values as a function of X and t. First
            dimension corresponds to time, second dimension corresponds to
            space.
        times: np.array
            Array of times associated with the first dimension of deltas
        xs: np.array
            Array of X values associated with the second dimension of deltas

    """

    times = sol.times
    w_t_func = UnivariateSpline(times, sol.w_t, s=0)
    dw_dt_func = w_t_func.derivative()

    psi_t_func = UnivariateSpline(times, sol.psi_t, s=0)
    dpsi_dt_func = psi_t_func.derivative()
    d2psi_dt2_func = dpsi_dt_func.derivative()

    xmin, xmax = x_range
    xs = np.arange(xmin, xmax, dx)
    ys = Y(xs)
    ys_func = UnivariateSpline(xs, ys, s=0)
    d2ydx2 = ys_func.derivative().derivative()
    d3ydx3 = d2ydx2.derivative()


    del_dot_term = ((xs*d3ydx3(xs) + 3.0*d2ydx2(xs))*
        dw_dt_func(plot_t)/w_t_func(plot_t))
    psi_dot_term = d2ydx2(xs)*d2psi_dt2_func(plot_t)/dpsi_dt_func(plot_t)
    nu_value = (
        d2ydx2(xs)*nu(psi_t_func(plot_t), poloidal_mode, lundquist_number, r_s)
    )

    #fig_derivs, ax_derivs = plt.subplots(1)
    #ax_derivs.plot(xs, xs*d3ydx3(xs)+3.0*d2ydx2(xs))

    fig, ax = plt.subplots(1, figsize=(4,3))

    ax.plot(
        xs, abs(nu_value+psi_dot_term),
        label=r"$|\nu Y'' + Y'' \delta\ddot{\psi}/\delta\dot{\psi}|$"
    )
    #ax.plot(
    #    times, psi_dot_term,
    #    label=r"$|Y'' \delta\ddot{\psi}/\delta\dot{\psi}|$"
    #)
    ax.plot(
        xs, abs(del_dot_term),
        label=r"$|[XY''' + 3Y''] \dot{\delta}/\delta|$"
    )

    ax.set_xlabel(r"X")
    ax.set_ylabel(r"Contribution to $Y'' \delta^4_{ql}(X, t)$")

    ax.set_title(r"$\bar{\omega}_A t$"f"={plot_t:.1f}")

    ax.legend(prop={'size':8})

    #ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()

    savefig(f"delql_contributions_t={plot_t:.2f}")

def compare_delql_terms():
    """
    Plot the full quasi-linear layer width as a function of (X, t) on a heatmap.
    """
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "./output/18-08-2023_16:41_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

    ts_to_plot = [100.0, 1e3, 1e5, 1e6]
    for t in ts_to_plot:
        plot_delql_terms(
            ql_sol,
            m,n,
            S,
            r_s,
            (-100, 100),
            t
        )



if __name__=='__main__':
    #constant_psi_approx()
    #convergence_of_delta_prime()
    #constant_psi_approx()
    compare_delql_terms()

    plt.show()
