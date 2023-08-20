import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from helpers import TimeDependentSolution
from y_sol import Y
from helpers import classFromArgs

def F(ql_sol: TimeDependentSolution,
      toroidal_mode: int,
      mag_shear: float):
    return -ql_sol.dpsi_dt / (toroidal_mode*mag_shear*ql_sol.w_t)


def potential(ql_sol: TimeDependentSolution,
              toroidal_mode: int,
              mag_shear: float,
              xs: np.array):
    f =  F(ql_sol, toroidal_mode, mag_shear)

    Xs = np.outer(xs, 1.0/ql_sol.w_t)
    phi = Y(Xs) * f

    return phi


def check_solution_is_valid(phi: np.ndarray,
                            xs: np.array,
                            times: np.array,
                            dpsi_dt: np.array,
                            delta_t: np.array,
                            toroidal_mode: int,
                            mag_shear: float):
    n = toroidal_mode
    s = mag_shear

    dphi_dx = np.gradient(phi, xs, axis=0, edge_order=2)
    d2phi_dx2 = np.gradient(dphi_dx, xs, axis=0, edge_order=2)

    #TODO: Multiply everything by x^2 to avoid singularity

    d2phi_term = np.outer(d2phi_dx2, delta_t**4)
    psi_term = np.outer(xs/(n*s), dpsi_dt)    
    phi_term = phi*(xs**2)

    #d4_x2 = np.outer(1.0/xs**2, delta_t**4)
    #psi_term = np.outer(1.0/(n*s*xs), dpsi_dt)

    #print(d4_x2.shape)
    #print(psi_term.shape)

    diff = (d2phi_term - phi_term + psi_term)**2

    fig, ax = plt.subplots(1)
    ax.imshow(
        diff,
        extent=[min(times), max(times), min(xs), max(xs)]
    )
    ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))

def potential_from_data():
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "./output/19-08-2023_14:29_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

    xs = np.arange(-0.025, 0.025, 0.005)
    phi = potential(ql_sol, n, s, xs)

    fig, ax = plt.subplots(1)
    times = ql_sol.times
    ax.imshow(
        phi,
        extent=[min(times), max(times), min(xs), max(xs)]
    )
    ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))

    check_solution_is_valid(
        phi,
        xs,
        times,
        ql_sol.dpsi_dt,
        ql_sol.w_t,
        n,
        s
    )

if __name__=='__main__':
    potential_from_data()
    plt.show()

