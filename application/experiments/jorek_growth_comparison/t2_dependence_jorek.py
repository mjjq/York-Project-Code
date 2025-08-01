from scipy.interpolate import CloughTocher2DInterpolator, UnivariateSpline
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import medfilt

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
    
    # dpsi_spline = UnivariateSpline(t_vals, delta_psi_vals)
    # dpsi_dt_spline = dpsi_spline.derivative()

    # dpsi_dt_vals = dpsi_dt_spline(t_vals)

    d_delta_psi = np.diff(delta_psi_vals)
    d_t = np.diff(t_vals)

    dpsi_dt_vals = d_delta_psi/d_t

    comparison_index = int(0.05*len(delta_psi_vals))
    dpsi_dt_ratio = np.array([
        dpsi_dt_vals[i+comparison_index]/dpsi_dt_vals[i] 
        for i in range(len(dpsi_dt_vals)-comparison_index)
    ])

    xi_ratio = np.array([
        np.sqrt(delta_psi_vals[i+comparison_index]/delta_psi_vals[i])
        for i in range(len(delta_psi_vals)-comparison_index)
    ])

    diff = np.abs(dpsi_dt_ratio - xi_ratio[:-1])

    fig, ax = plt.subplots(1)
    ax.plot(t_vals[:-(comparison_index+1)], diff)
    ax.set_yscale('log')

    t_0_arg = np.argmin(diff)
    t_1_arg = t_0_arg + comparison_index

    return t_vals[t_0_arg], t_vals[t_1_arg]


def quadratic(t, t_0, delta_psi_0, c): 
    return ((t-t_0 + 2*c*delta_psi_0**0.5)/(2*c))**2


def fit_quadratic(t_vals: np.array,
                  delta_psi_vals: np.array):
    t_vals = t_vals/np.max(t_vals)
    initialisation_t_cutoff = 0.3
    saturation_t_cutoff = 0.9

    cutoff_filter = (t_vals>initialisation_t_cutoff) & (t_vals<saturation_t_cutoff)

    t_0,t_1 = find_best_quad_fit_location(
        t_vals[cutoff_filter], 
        delta_psi_vals[cutoff_filter]
    )
    print(t_0, t_1)

    dpsi_0 = np.interp(t_0, t_vals, delta_psi_vals)

    fit_filter = (t_vals>=t_0) & (t_vals<t_1)

    t_vals_filt = t_vals[fit_filter]
    dpsi_vals_filt = delta_psi_vals[fit_filter]

    fig_f, ax_f = plt.subplots(1)
    ax_f.plot(t_vals_filt, dpsi_vals_filt)
    ax_f.scatter([t_0], [dpsi_0])

    partial_quadratic = lambda t, c : quadratic(t, t_0, dpsi_0, c)

    popt, pcov = curve_fit(
        partial_quadratic, 
        t_vals_filt, 
        dpsi_vals_filt,
        p0=[5.0]
    )
    c_fit = popt[0]
    c_std = np.sqrt(np.diag(pcov))[0]
    t2_results = partial_quadratic(t_vals, *popt)
    t2_results_min = partial_quadratic(t_vals, c_fit-c_std)
    t2_results_max = partial_quadratic(t_vals, c_fit+c_std)
    
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
    ax.scatter(
        [t_0], [dpsi_0], marker='x', color='red'
    )

    #ax.set_xscale('log')
    ax.set_yscale('log')

    dpsi_plot_min = 0.1*dpsi_0
    t_plot_min = t_vals[np.argmin(np.abs(dpsi_plot_min-delta_psi_vals))]

    ax.set_xlim(left=t_plot_min, right=1.05)
    ax.set_ylim(bottom=dpsi_plot_min, top=1.1*max(delta_psi_vals))

    ax.set_xlabel(r"Time ($t_{max}$)")
    ax.set_ylabel(r"$\delta\psi$ (arb)")
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

    fit_quadratic(t_vals, delta_psi_vals)

    fig, ax = plt.subplots(1)
    ax.plot(t_vals, delta_psi_vals)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()
