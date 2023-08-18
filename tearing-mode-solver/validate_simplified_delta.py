import numpy as np
from dataclasses import dataclass, fields
import pandas as pd
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt

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

    dpsi_dt_func = UnivariateSpline(times, sol.dpsi_dt)
    d2psi_dt2_func = dpsi_dt_func.derivative()

    xs = np.linspace(-100.0, 100.0, 10000)
    ys = Y(xs)
    ys_func = UnivariateSpline(xs, ys, s=0)
    d2ydx2 = ys_func.derivative().derivative()
    d3ydx3 = d2ydx2.derivative()

    del_dot_term = dw_dt_func(times)/w_t_func(times)
    psi_dot_term = d2psi_dt2_func(times)/dpsi_dt_func(times)
    nu_value = nu(sol.psi_t, poloidal_mode, lundquist_number, r_s)

    pre_factor = 1.0/( lundquist_number * (toroidal_mode*mag_shear)**2)

    deltas = []

    for x in xs:
        M = x*d3ydx3(x)/d2ydx2(x)
        del_dot = - del_dot_term * (3.0+M)
        psi_dot = psi_dot_term

        delta_pow_4 = pre_factor*(nu_value + psi_dot + del_dot)

        delta_pow_4[delta_pow_4<0.0] = 0.0

        delta_at_x = delta_pow_4**(1/4)

        deltas.append(delta_at_x)

    deltas = np.array(deltas)

    return deltas, times, xs

def delta_prime_full(sol: QuasiLinearSolution):
    return

def classFromArgs(className, df):
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {col : np.array(df[col]) for col in df.columns 
                       if col in fieldSet}
    return className(**filteredArgDict)

if __name__=='__main__':
    fname = "./output/17-08-2023_18:59_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(QuasiLinearSolution, df)

    delqls, times, xs = del_ql_full(ql_sol, 2, 1, 1e8, 10.0, 0.79)
    #delqls = delqls[:,::1000]
    
    fig, ax = plt.subplots(1, figsize=(4,4))
    plt.imshow(
        delqls,
        extent=[min(times), max(times), min(xs), max(xs)]
    )
    
    ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))
    
    fig2, ax2 = plt.subplots(1, figsize=(4,3))
    
    lineout = delqls[:,1]
    ax2.plot(xs, lineout)
