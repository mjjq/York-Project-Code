import numpy as np
from dataclasses import dataclass, fields
import pandas as pd
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from scipy.integrate import quad, simpson
from tqdm import tqdm, trange

from y_sol import Y
from new_ql_solver import QuasiLinearSolution, nu, layer_width

def del_ql_full(sol: QuasiLinearSolution,
                toroidal_mode: int,
                poloidal_mode: int,
                lundquist_number: float,
                mag_shear: float,
                r_s: float):

    times = sol.times
    w_t_func = UnivariateSpline(times, sol.w_t, s=0)
    dw_dt_func = w_t_func.derivative()

    dpsi_dt_func = UnivariateSpline(times, sol.dpsi_dt, s=0)
    d2psi_dt2_func = UnivariateSpline(times, sol.d2psi_dt2,s=0)#dpsi_dt_func.derivative()

    xs = np.linspace(-20.0, 20.0, 1000)
    ys = Y(xs)
    ys_func = UnivariateSpline(xs, ys, s=0)
    d2ydx2 = ys_func.derivative().derivative()
    d3ydx3 = d2ydx2.derivative()

    del_dot_term = dw_dt_func(times)/w_t_func(times)
    psi_dot_term = d2psi_dt2_func(times)/dpsi_dt_func(times)
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
                     r_s: float,
                     lundquist_number: float):
    
    ys = Y(xs)
    
    
    delta_primes = []
    tqdm_range = trange(len(times), leave=True)
    for i in tqdm_range:
        t= times[i]
        delta_ql_x = delta_qls[:,i]
        
        int_result = simpson(
            (1.0+xs*ys)*delta_ql_x, x=xs
        )

        psi = psi_t[i]
        dpsi = dpsi_dt[i]
        
        delta_primes.append(lundquist_number*dpsi*int_result/(psi*r_s))
    
    return delta_primes

def classFromArgs(className, df):
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {col : np.array(df[col]) for col in df.columns 
                       if col in fieldSet}
    return className(**filteredArgDict)

if __name__=='__main__':
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401
    
    fname = "./output/18-08-2023_16:41_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(QuasiLinearSolution, df)

    simple_integration()

    delqls, times, xs = del_ql_full(ql_sol, m, n, S, s, r_s)
    #delqls = delqls[:,::1000]
    
    # delta_primes = delta_prime_full(
    #     delqls,
    #     xs,
    #     times,
    #     ql_sol.psi_t,
    #     ql_sol.dpsi_dt,
    #     r_s,
    #     S
    # )
    
    # fig_dp, ax_dp = plt.subplots(1, figsize=(4,4))
    # ax_dp.plot(times, delta_primes, label=r"Exact $\Delta'$")
    # ax_dp.plot(times, ql_sol.delta_primes, label=r"Approximate $\Delta'$")
    
    # fig, ax = plt.subplots(1, figsize=(4,4))
    # plt.imshow(
    #     delqls,
    #     extent=[min(times), max(times), min(xs), max(xs)]
    # )
    
    # ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))
    
    fig2, ax2 = plt.subplots(1, figsize=(4,3))
    
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
        
    plt.show()
