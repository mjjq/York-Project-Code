from scipy.interpolate import CloughTocher2DInterpolator, UnivariateSpline
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import medfilt

from tearing_mode_solver.helpers import savefig



def plot_reconnection_rate(t_vals: np.array,
                           delta_psi_vals: np.array):
    t_vals = t_vals/np.max(t_vals)
    saturation_filter = t_vals <= 0.99
    t_vals = t_vals[saturation_filter]
    delta_psi_vals = delta_psi_vals[saturation_filter]

    
    d_delta_psi = np.diff(delta_psi_vals)
    dt_vals = np.diff(t_vals)

    dpsi_dt = d_delta_psi/dt_vals
    
    fig, ax = plt.subplots(1, figsize=(4,3))
    ax.plot(
        t_vals,
        dpsi_dt,
        color='black'
    )

    ax.set_xlabel(r"Time ($t_{max}$)")
    ax.set_ylabel(r"$t_{max} d\delta\psi/dt$ (arb)")
    ax.grid()
    ax.legend()

    fig.tight_layout()

    #savefig("quadratic_fit")


if __name__=='__main__':
    import sys

    fname = sys.argv[1]

    data = np.loadtxt(fname, skiprows=1)

    t_vals = data[:,0]
    delta_psi_vals = data[:,2]**0.5

    plot_reconnection_rate(t_vals, delta_psi_vals)

    # fig, ax = plt.subplots(1)
    # ax.plot(t_vals, delta_psi_vals)
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    plt.show()
