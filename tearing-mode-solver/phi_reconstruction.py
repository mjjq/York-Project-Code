import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

from helpers import TimeDependentSolution
from y_sol import Y
from helpers import classFromArgs, savefig
from new_ql_solver import mode_width

def F(ql_sol: TimeDependentSolution,
      toroidal_mode: int,
      mag_shear: float) -> np.array:
    """
    The time-dependent component of the separation of variables function.

    I.e., for the potential phi(x,t) = Y(X)F(t), this function returns F(t)

    Parameters
        ql_sol: TimeDependentSolution
            Quasi-linear time-dependent tearing mode solution (calculated
            either from the gamma or delta models)
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        mag_shear: float
            Magnetic shear at the resonant surface.
    """
    return -ql_sol.dpsi_dt / (toroidal_mode*mag_shear*ql_sol.w_t)


def potential(ql_sol: TimeDependentSolution,
              toroidal_mode: int,
              mag_shear: float,
              xs: np.array) -> np.ndarray:
    """
    Re-construct the electric potential from the quasi-linear time-dependent
    solution.

    Parameters
        ql_sol: TimeDependentSolution
            Quasi-linear time-dependent tearing mode solution (calculated
            either from the gamma or delta models)
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        mag_shear: float
            Magnetic shear at the resonant surface.
        xs: np.array
            Spatial values over which the potential must be calculated.
    """
    f =  F(ql_sol, toroidal_mode, mag_shear)

    # Calculate outer product between the x-array and island widths. Equivalent
    # to X = x/delta(t)
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
    """
    Compute the differential equation governing the time-dependence of the
    perturbed flux from a reconstructed electric potential. Use this to
    determine if the electric potential satisfies the differential equation.

    Parameters
        phi: np.ndarray
            The electric potential. (2D array, where first dimension corresponds
            to time and second dimension to space).
        xs: np.array:
            Spatial values over which the potential has been calculated.
        times: np.array:
            Temporal values over which the potential has been calculated
        dpsi_dt: np.array
            The first time derivative in the perturbed flux. Must have same
            dimension as times
        toroidal_mode: int
            Toroidal mode number of the tearing mode
        mag_shear: float
            Magnetic shear at the resonant surface.

    """
    n = toroidal_mode
    s = mag_shear

    dphi_dx = np.gradient(phi, xs, axis=0, edge_order=2)
    d2phi_dx2 = np.gradient(dphi_dx, xs, axis=0, edge_order=2)

    
    d2phi_term = ((d2phi_dx2.T/xs**2).T) * delta_t**4 / phi
    psi_term = np.outer(1.0/(xs*n*s), dpsi_dt)   / phi  
    #phi_term = (phi.T*xs**2).T / phi

    #d4_x2 = np.outer(1.0/xs**2, delta_t**4)
    #psi_term = np.outer(1.0/(n*s*xs), dpsi_dt)

    #print(d4_x2.shape)
    #print(psi_term.shape)

    diff = (d2phi_term + psi_term)**2
    
    diff[len(xs)//2] = np.ones(len(times))

    fig, ax = plt.subplots(1, figsize=(4,3))
    im = ax.imshow(
        diff,
        extent=[min(times), max(times), min(xs), max(xs)],
        vmin=0.5, vmax=1.5
    )
    fig.colorbar(im)
    ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))
    
    ax.set_xlabel(r"Normalised time ($1/\bar{\omega}_A$)")
    ax.set_ylabel(r"x ($r_s$)")
    
    fig.tight_layout()

def potential_from_data():
    """
    Load pre-computed quasi-linear solution data, re-construct the electric
    potential, then plot the function as a heatmap.
    """
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "./output/29-08-2023_10:53_new_ql_tm_time_evo_(m,n,A,q0)=(2,1,1e-10,1.0).csv"
    df = pd.read_csv(fname)
    df = df.iloc[:100000:1000,:]
    ql_sol = classFromArgs(TimeDependentSolution, df)
    ql_sol.w_t = mode_width(
        ql_sol.psi_t,
        ql_sol.dpsi_dt,
        ql_sol.d2psi_dt2,
        r_s,
        m,
        n,
        s,
        S
    )

    xs = np.linspace(-0.5, 0.5, 1000)
    #print(xs)
    phi = potential(ql_sol, n, s, xs)

    fig, ax = plt.subplots(1, figsize=(4,3))
    times = ql_sol.times
    im = ax.imshow(
        phi,
        extent=[min(times), max(times), min(xs), max(xs)]
    )

    fig.colorbar(im)
    ax.set_aspect((max(times)-min(times))/(max(xs)-min(xs)))
    
    ax.set_xlabel(r"Normalised time ($1/\bar{\omega}_A$)")
    ax.set_ylabel(r"x ($r_s$)")
    
    fig.tight_layout()
    
    orig_fname, ext = os.path.splitext(os.path.basename(fname))
    savefig(f"potential_rec_{orig_fname}")

    check_solution_is_valid(
        phi,
        xs,
        times,
        ql_sol.dpsi_dt,
        ql_sol.w_t,
        n,
        s
    )
    
    orig_fname, ext = os.path.splitext(os.path.basename(fname))
    savefig(f"pde_validation_{orig_fname}")

if __name__=='__main__':
    potential_from_data()
    plt.show()

