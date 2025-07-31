from scipy.interpolate import CloughTocher2DInterpolator, UnivariateSpline
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from tearing_mode_solver.helpers import savefig


def find_best_quad_fit_location(t_vals: np.array,
                                delta_psi_vals: np.array) -> float:
    """
    If function is quadratic, it must satisfy the equation
    dx/dt = A*sqrt(x)

    Hence, this is true in the discrete form via

    dx_i/dt = A*sqrt(x_i)

    and

    dx_{i+1}/dt = A*sqrt(x_{i+1}), implying

    (dx_{i+1}/dt) / (dx_i/dt) = sqrt(x_{i+1}/x_i)

    Hence, find the value of x_i that minimises the error in the above.
    This is what we do here
    """
    
    dpsi_spline = UnivariateSpline(t_vals, delta_psi_vals)
    dpsi_dt_spline = dpsi_spline.derivative()

    dpsi_dt_vals = dpsi_dt_spline(t_vals)

    dpsi_dt_ratio = np.array([
        dpsi_dt_vals[i+1]/dpsi_dt_vals[i] 
        for i in range(len(dpsi_dt_vals)-1)
    ])

    xi_ratio = np.array([
        np.sqrt(delta_psi_vals[i+1]/delta_psi_vals[i])
        for i in range(len(delta_psi_vals)-1)
    ])

    diff = np.abs(dpsi_dt_ratio - xi_ratio)

    fig, ax = plt.subplots(1)
    ax.plot(t_vals[:-1], diff)
    ax.set_yscale('log')

    return t_vals[np.argmin(diff)]


def quadratic(t, t_0, delta_psi_0, c): 
    return ((t-t_0 + 2*c*delta_psi_0**0.5)/(2*c))**2


def fit_quadratic(t_vals: np.array,
                  delta_psi_vals: np.array):
    t_vals = t_vals/np.max(t_vals)
    t_0 = find_best_quad_fit_location(t_vals, delta_psi_vals)
    print(t_0)

    dpsi_0 = np.interp(t_0, t_vals, delta_psi_vals)

    t_vals_filt = t_vals[(t_vals>=t_0)]
    dpsi_vals_filt = delta_psi_vals[(t_vals>=t_0)]

    partial_quadratic = lambda t, c : quadratic(t, t_0, dpsi_0, c)

    popt, pcov = curve_fit(
        partial_quadratic, 
        t_vals_filt, 
        dpsi_vals_filt,
        bounds=([0.0], [np.inf]),
        p0=[1.0],
        method='trf'
    )
    c_fit = popt[0]
    c_std = np.sqrt(np.diag(pcov))[0]
    t2_results = partial_quadratic(t_vals, *popt)
    t2_results_min = partial_quadratic(t_vals, c_fit-c_std)
    t2_results_max = partial_quadratic(t_vals, c_fit+c_std)

    print(c_fit)
    
    fig, ax = plt.subplots(1, figsize=(4,3))
    ax.plot(
        t_vals,
        delta_psi_vals,
        color='black',
        label='JOREK'
    )
    ax.plot(
        t_vals, 
        t2_results, 
        linestyle='--', color='red',
        label=r"$t^2$ fit: "f"C={c_fit:.3f}"
    )

    #ax.set_xscale('log')
    ax.set_yscale('log')

    dpsi_plot_min = 0.1*max(delta_psi_vals)
    t_plot_min = t_vals[np.argmin(np.abs(dpsi_plot_min-delta_psi_vals))]

    ax.set_xlim(left=t_plot_min, right=1.05)
    ax.set_ylim(bottom=dpsi_plot_min, top=1.1*max(delta_psi_vals))

    ax.set_xlabel(r"Time ($t_{max}$)")
    ax.set_ylabel(r"$\delta\psi$ (arb)")
    ax.grid()
    ax.legend()

    fig.tight_layout()

    savefig("quadratic_fit")


if __name__=='__main__':
    import sys

    fname = sys.argv[1]

    data = np.loadtxt(fname, skiprows=1)

    t_vals = data[:,0]
    delta_psi_vals = data[:,2]**0.5

    fit_quadratic(t_vals, delta_psi_vals)

    fig, ax = plt.subplots(1)
    ax.plot(t_vals, delta_psi_vals)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()
