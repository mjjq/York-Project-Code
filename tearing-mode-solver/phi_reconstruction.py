import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from new_ql_solver import QuasiLinearSolution
from y_sol import Y
from pyplot_helper import classFromArgs

def F(ql_sol: QuasiLinearSolution,
      toroidal_mode: int,
      mag_shear: float):
    return -ql_sol.dpsi_dt / (toroidal_mode*mag_shear*ql_sol.w_t)


def potential(ql_sol: QuasiLinearSolution,
              toroidal_mode: int,
              mag_shear: float,
              xs: np.array):
    f =  F(ql_sol, toroidal_mode, mag_shear)

    Xs = np.outer(xs, 1.0/ql_sol.w_t)
    phi = Y(Xs) * f

    return phi

def potential_from_data():
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "./output/18-08-2023_16:41_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(QuasiLinearSolution, df)

    xs = np.arange(-1.0, 1.0, 0.1)
    phi = potential(ql_sol, n, s, xs)

    fig, ax = plt.subplots(1)
    times = ql_sol.times
    ax.imshow(
        phi,
        extent=[min(times), max(times), min(xs), max(xs)]
    )
    ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))

if __name__=='__main__':
    potential_from_data()
    plt.show()

