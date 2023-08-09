from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize_scalar
from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from pyplot_helper import savefig

def dj_dr(radial_coordinate: float,
          shaping_exponent: float = 2.0) -> float:
    """
    Normalised derivative in the current profile.

    Current is normalised to the on-axis current J_0.
    radial_coordinate is normalised to the minor radius a.
    """
    r = radial_coordinate
    nu = shaping_exponent

    return -2*nu*(r) * (1-r**2)**(nu-1)

@np.vectorize
def q(radial_coordinate: float,
      shaping_exponent: float = 2.0) -> float:
    r = radial_coordinate
    nu = shaping_exponent
    
    # Prevent division by zero for small r values.
    # For this function, in the limit r->0, q(r)->1. This is proven
    # mathematically in the lab book.
    #print(r)
    if np.abs(r) < 1e-5:
        return 1.0

    return (nu+1)*(r**2)/(1-(1-r**2)**(nu+1))

def rational_surface(target_q: float,
                     shaping_exponent: float = 2.0) -> float:
    """
    Compute the location of the rational surface of the q-profile defined in q().
    """
    
    # Establish the function to pass to scipy's scalar minimiser. We want to 
    # find the value of r such that q(r) = target_q. This is equivalent to
    # finding the value of r that minimises (q(r) - target_q)^2
    # Call vectorize so that it converts q(r) into a function that can take
    # np arrays. This is necessary because q(r) contains an if statement which
    # fails if the function is not vectorized.
    fun = np.vectorize(lambda r : (q(r, shaping_exponent) - target_q)**2)
    
    rs = minimize_scalar(fun, bounds=(0.0, 1.0), method='bounded')

    return rs.x


def compute_derivatives(y: Tuple[float, float],
                        r: float,
                        poloidal_mode: int,
                        toroidal_mode: int,
                        j_profile_derivative,
                        q_profile,
                        axis_q: float,
                        epsilon: float = 1e-5) -> Tuple[float, float]:
    """
    Compute derivatives for the perturbed flux outside of the resonant region
    (resistivity and inertia neglected).
    
    The equation we are solving is
    
    r^2 f'' + rf' + 2rfA - m^2 f = 0,
    
    where f is the normalised perturbed flux, r is the radial co-ordinate,
    m is the poloidal mode number, and A = (q*m*dj_dr)/(n*q_0*q - m), where
    q is the normalised safety factor, dj_dr is the normalised gradient in
    the toroidal current, q_0 is the normalised safety factor at r=0, and
    n is the toroidal mode number.
    
    The equation is singular at r=0, so to get ODEINT to play nicely we
    formulate the following coupled ODE:
        
    y = [f, r^2 f']
         
    y'= [f', r^2 f'' + 2rf']
    
      = [f', f(m^2 - 2rA) + rf']
      
    Then, for r=0, 
    
    y'= [f', m^2 f]

    We set the initial f' at r=0 to some arbitrary positive value. Note that,
    for r>0, this function will then receive a tuple containing f and 
    **r^2 f'** instead of just f and f', as this is what we are calculating
    in y as defined above.
    
    Hence, for r>0, we must divide the incoming f' by r^2.

    Parameters
    ----------
    y : Tuple[float, float]
        Perturbed flux and radial derivative in perturbed flux.
    r : float
        Radial co-ordinate.
    poloidal_mode : int
        Poloidal mode number.
    toroidal_mode : int
        Toroidal mode number.
    j_profile_derivative : func
        Derivative in the current profile. Must be a function which accepts
        the radial co-ordinate r as a parameter.
    q_profile : TYPE
        Safety factor profile. Must be a function which accepts the radial
        co-ordinate r as a parameter.
    axis_q : float, optional
        Value of the normalised safety factor on-axis. The default is 1.0.
    epsilon : float, optional
        Tolerance value to determine values of r which are sufficiently close
        to r=0. The default is 1e-5.

    Returns
    -------
    Tuple[float, float]
        Radial derivative in perturbed flux and second radial derivative in 
        perturbed flux.

    """
    psi, dpsi_dr = y
    
    if np.abs(r) > epsilon:
        dpsi_dr = dpsi_dr/r**2

    m = poloidal_mode
    n = toroidal_mode
    q_0 = axis_q
    
    if np.abs(r) < epsilon:
        d2psi_dr2 = psi*m**2
    else:
        dj_dr = j_profile_derivative(r)
        q = q_profile(r)
        A = (q*m*dj_dr)/(n*q_0*q - m)
        d2psi_dr2 = psi*(m**2 - 2*A*r) + r*dpsi_dr
        
    #print(dpsi_dr)

    return dpsi_dr, d2psi_dr2

@dataclass
class TearingModeSolution():
    # All quantities in this class are normalised
    
    # Perturbed flux and derivative starting from the poloidal axis
    psi_forwards: np.array
    dpsi_dr_forwards: np.array
    
    # Radial domain for the forward solution
    r_range_fwd: np.array
    
    # Perturbed flux and derivative starting from the edge of the plasma
    # going inwards
    psi_backwards: np.array
    dpsi_dr_backwards: np.array
    
    # Radial domain for the backward solution
    r_range_bkwd: np.array
    
    # Location of the resonant surface
    r_s: float
    
def scale_tm_solution(tm: TearingModeSolution, scale_factor: float)\
    -> TearingModeSolution:
    return TearingModeSolution(
        tm.psi_forwards*scale_factor, 
        tm.dpsi_dr_forwards*scale_factor, 
        tm.r_range_fwd, 
        tm.psi_backwards*scale_factor, 
        tm.dpsi_dr_backwards*scale_factor, 
        tm.r_range_bkwd, 
        tm.r_s
    )

def solve_system(poloidal_mode: int, 
                 toroidal_mode: int, 
                 axis_q: float,
                 n: int = 100000) -> TearingModeSolution:
    """
    Generate solution for peturbed flux over the minor radius of a cylindrical
    plasma given the mode numbers of the tearing mode.

    Parameters
    ----------
    poloidal_mode : int
        Poloidal mode number.
    toroidal_mode : int
        Toroidal mode number.
    axis_q : float, optional
        Value of the safety factor on-axis. The default is 1.0.
    n: int, optional
        Number of elements in the integrand each for the forwards and 
        backwards solutions. The default is 10000.

    Returns
    -------
    TearingModeSolution:
        Quantities relating to the tearing mode solution.

    """
    initial_psi = 0.0
    initial_dpsi_dr = 1.0

    q_rs = poloidal_mode/(toroidal_mode*axis_q)
    if q_rs >= q(1.0) or q_rs <= q(0.0):
        raise ValueError("Rational surface located outside bounds")

    r_s = rational_surface(poloidal_mode/(toroidal_mode*axis_q))

    r_s_thickness = 0.0001

    #print(f"Rational surface located at r={r_s:.4f}")

    # Solve from axis moving outwards towards rational surface
    r_range_fwd = np.linspace(0.0, r_s-r_s_thickness, n)

    results_forwards = odeint(
        compute_derivatives,
        (initial_psi, initial_dpsi_dr),
        r_range_fwd,
        args = (poloidal_mode, toroidal_mode, dj_dr, q, axis_q),
        tcrit=(0.0)
    )

    psi_forwards, dpsi_dr_forwards = (
        results_forwards[:,0], results_forwards[:,1]
    )

    # Solve from minor radius moving inwards towards rational surface
    r_range_bkwd = np.linspace(1.0, r_s+r_s_thickness, n)

    results_backwards = odeint(
        compute_derivatives,
        (initial_psi, -initial_dpsi_dr),
        r_range_bkwd,
        args = (poloidal_mode, toroidal_mode, dj_dr, q, axis_q)
    )

    psi_backwards, dpsi_dr_backwards = (
        results_backwards[:,0], results_backwards[:,1]
    )
    #print(psi_backwards)
    #print(dpsi_dr_backwards)
    
    # Rescale the forwards solution such that its value at the resonant
    # surface matches the psi of the backwards solution. This is equivalent
    # to fixing the initial values of the derivatives such that the above
    # relation is satisfied
    fwd_res_surface = psi_forwards[-1]
    bkwd_res_surface = psi_backwards[-1]
    psi_forwards = psi_forwards * bkwd_res_surface/fwd_res_surface
    dpsi_dr_forwards = dpsi_dr_forwards * bkwd_res_surface/fwd_res_surface
    
    #print(dpsi_dr_forwards)
    #print(r_range_fwd)
    
    # Recover original derivatives as the compute_derivatives function returns
    # r^2 * f' for r>0. For r=0, the compute_derivatives function returns f'
    # so no need to divide by r^2.
    dpsi_dr_forwards[1:] = dpsi_dr_forwards[1:]/(r_range_fwd[1:]**2)
    dpsi_dr_backwards = dpsi_dr_backwards/(r_range_bkwd**2)
    
    #print(dpsi_dr_forwards)
    
    return TearingModeSolution(
        psi_forwards, dpsi_dr_forwards, r_range_fwd,
        psi_backwards, dpsi_dr_backwards, r_range_bkwd,
        r_s
    )
    
    # return psi_forwards, dpsi_dr_forwards, r_range_fwd, \
    #     psi_backwards, dpsi_dr_backwards, r_range_bkwd , r_s
    
    
def delta_prime(tm_sol: TearingModeSolution,
                epsilon: float = 1e-10):
    psi_plus = tm_sol.psi_backwards[-1]
    psi_minus = tm_sol.psi_forwards[-1]
    
    if abs(psi_plus - psi_minus) > epsilon:
        raise ValueError(
            f"""Forwards and backward solutions 
            should be equal at resonant surface.
            (psi_plus={psi_plus}, psi_minus={psi_minus})."""
        )
    
    dpsi_dr_plus = tm_sol.dpsi_dr_backwards[-1]
    dpsi_dr_minus = tm_sol.dpsi_dr_forwards[-1]
    
    return (dpsi_dr_plus - dpsi_dr_minus)/psi_plus

def solve_and_plot_system_simple():
    poloidal_mode = 3
    toroidal_mode = 3
    axis_q = 0.999

    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)

    delta_p = delta_prime(tm)

    print(f"Delta prime = {delta_p}")

    fig, ax = plt.subplots(1, figsize=(4,3), sharex=True)

    #ax4.plot(r_range_fwd, dpsi_dr_forwards)
    #ax4.plot(r_range_bkwd, dpsi_dr_backwards)

    ax.plot(
        tm.r_range_fwd, tm.psi_forwards, label='Solution below $\hat{r}_s$'
    )
    ax.plot(
        tm.r_range_bkwd, tm.psi_backwards, label='Solution above $\hat{r}_s$'
    )
    rs_line = ax.vlines(
         tm.r_s, ymin=0.0, ymax=np.max([tm.psi_forwards, tm.psi_backwards]),
         linestyle='--', color='red',
         label=f'Rational surface $\hat{{q}}(\hat{{r}}_s) = {poloidal_mode}/{toroidal_mode}$'
    )

    ax.set_xlabel("Normalised minor radial co-ordinate (r/a)")
    ax.set_ylabel(r"Normalised perturbed flux $\hat{\delta \psi}^{(1)}$")


    fig.tight_layout()

    plt.show()


def solve_and_plot_system():
    poloidal_mode = 3
    toroidal_mode = 3
    axis_q = 0.999
    
    tm = solve_system(poloidal_mode, toroidal_mode, axis_q)
    
    delta_p = delta_prime(tm)
    
    print(f"Delta prime = {delta_p}")

    fig, axs = plt.subplots(3, figsize=(6,10), sharex=True)
    ax, ax2, ax3 = axs
    
    #ax4.plot(r_range_fwd, dpsi_dr_forwards)
    #ax4.plot(r_range_bkwd, dpsi_dr_backwards)

    ax.plot(
        tm.r_range_fwd, tm.psi_forwards, label='Solution below $\hat{r}_s$'
    )
    ax.plot(
        tm.r_range_bkwd, tm.psi_backwards, label='Solution above $\hat{r}_s$'
    )
    rs_line = ax.vlines(
         tm.r_s, ymin=0.0, ymax=np.max([tm.psi_forwards, tm.psi_backwards]),
         linestyle='--', color='red', 
         label=f'Rational surface $\hat{{q}}(\hat{{r}}_s) = {poloidal_mode}/{toroidal_mode}$'
    )

    ax3.set_xlabel("Normalised minor radial co-ordinate (r/a)")
    ax.set_ylabel("Normalised perturbed flux ($\delta \psi / a^2 J_\phi$)")
    
    r = np.linspace(0, 1.0, 100)
    ax2.plot(r, dj_dr(r))
    ax2.set_ylabel("Normalised current gradient $[d\hat{J}_\phi/d\hat{r}]$")
    ax2.vlines(
        tm.r_s, 
        ymin=np.min(dj_dr(r)), 
        ymax=np.max(dj_dr(r)), 
        linestyle='--', 
        color='red'    
    )
    
    ax3.plot(r, np.vectorize(q)(r))
    ax3.set_ylabel("Normalised q-profile $[\hat{q}(\hat{r})]$")
    ax3.vlines(
        tm.r_s, 
        ymin=np.min(q(r)), 
        ymax=np.max(q(r)), 
        linestyle='--', 
        color='red'    
    )
    #ax3.set_yscale('log')
    ax.legend(prop={'size': 8})

    plt.show()




def magnetic_shear(resonant_surface: float,
                   poloidal_mode: int,
                   toroidal_mode: int) -> float:
    r_s = resonant_surface
    
    r = np.linspace(0, 1, 1000)
    dr = r[1]-r[0]
    q_values = q(r)
    dq_dr = np.gradient(q_values, dr)
    rs_id = np.abs(r_s - r).argmin()
    
    m = poloidal_mode
    n = toroidal_mode
    
    return (m/n)*r_s*dq_dr[rs_id]

def growth_rate_scale(lundquist_number: float,
                      r_s: float,
                      poloidal_mode: float,
                      toroidal_mode: float):
   
    # Equivalent to 2*pi*Gamma(3/4)/Gamma(1/4)
    gamma_scale_factor = 2.1236482729819393256107565
    
    m = poloidal_mode
    n = toroidal_mode
    S = lundquist_number
    
    s = magnetic_shear(r_s, m, n)
    
    grs = gamma_scale_factor**(-4/5)* r_s**(4/5) \
        * (n*s)**(2/5) / S**(3/5)
        
    return grs
    

def growth_rate(poloidal_mode: int,
                toroidal_mode: int,
                lundquist_number: float,
                axis_q: float = 1.0):
    
    m = poloidal_mode
    n = toroidal_mode
    S = lundquist_number
    
    try:
        tm = solve_system(m, n, axis_q)
    except ValueError:
        return np.nan, np.nan
    
    delta_p = delta_prime(tm)
    
    grs = growth_rate_scale(S, tm.r_s, m, n)

    growth_rate = grs*complex(delta_p)**(4/5)

    return delta_p, growth_rate.real

def layer_width(poloidal_mode: int,
                toroidal_mode: int,
                lundquist_number: float,
                axis_q: float = 1.0):
    m = poloidal_mode
    n = toroidal_mode
    S = lundquist_number

    try:
        tm = solve_system(m, n, axis_q)
    except ValueError:
        return np.nan

    delta_p, gr = growth_rate(m, n, S, axis_q)

    s = magnetic_shear(tm.r_s, m, n)

    return tm.r_s*(gr/(S*(n*s)**2))**(1/4)

def alfven_frequency_STEP():
    # 1.35e6 Hz for STEP, see lab book "STEP parameters" log
    return 1.35e6

def ps_correction(alfven_freq: float,
                  poloidal_mode: int,
                  toroidal_mode: int):
    """
    Pfirsch schluter correction to the alfven frequency at
    resonant surface.
    """

    return alfven_freq / (1+2*poloidal_mode/toroidal_mode)

def growth_rate_vs_mode_number():
    modes = [
        (2,1),
        (3,2),
        (4,2),
        (4,3),
        (5,2),
        (5,3),
        (5,4),
        (6,3),
        (6,4),
        (6,5)
    ]
    lundquist_number = 1e8
    
    fig, ax = plt.subplots(1)
    
    results = []

    alfven_frequency = alfven_frequency_STEP()
    
    for m,n in modes:
        delta_p, gr = growth_rate(m,n,lundquist_number)

        alfven_corrected = ps_correction(alfven_frequency, m, n)

        abs_growth = gr*alfven_corrected

        linear_layer_width = layer_width(m,n,lundquist_number)

        results.append(
            (delta_p, gr, alfven_corrected, abs_growth, linear_layer_width)
        )
        
    print(f"mode & delta_p & gamma/omega & omega_bar & gamma & lin_growth")
    for i,mode in enumerate(modes):
        m, n = mode
        (delta_p, growth, af_corrected,
         abs_growth, linear_layer_width) = results[i]
        print(
            f"{m} & {n} & {delta_p:.2f} & ${growth:.2e}$ & "
            f"${af_corrected:.2e}$ & ${abs_growth:.2e}$ & "
            f"${linear_layer_width:.2e}$"+r"\\")

def test_gradient():
    m=3
    n=2
    lundquist_number = 1e8
    axis_q = 1.0
    
    tm = solve_system(m, n, axis_q)
    
    dr_fwd = tm.r_range_fwd[-1] - tm.r_range_fwd[-2]
    dpsi_dr_fwd = np.gradient(tm.psi_forwards, dr_fwd)
    
    dr_bkwd = tm.r_range_bkwd[-1] - tm.r_range_bkwd[-2]
    dpsi_dr_bkwd = np.gradient(tm.psi_backwards, dr_bkwd)
    
    print(dpsi_dr_fwd[-1], tm.dpsi_dr_forwards[-1])
    print(dpsi_dr_bkwd[-1], tm.dpsi_dr_backwards[-1])
    
    fig, axs = plt.subplots(2)
    
    ax, ax2 = axs
    
    ax.plot(
        tm.r_range_fwd[-10:], dpsi_dr_fwd[-10:], label='np.gradient'
    )
    ax.plot(
        tm.r_range_fwd[-10:], tm.dpsi_dr_forwards[-10:], label='ODEINT gradient',
        color='black'
    )
    
    ax2.plot(
        tm.r_range_bkwd[-10:], dpsi_dr_bkwd[-10:], label='np.gradient'
    )
    ax2.plot(
        tm.r_range_bkwd[-10:], tm.dpsi_dr_backwards[-10:], label='ODEINT gradient',
        color='black'
    )
    
    
    dpsi_dr_fwd = np.gradient(tm.psi_forwards, dr_fwd, edge_order=2)
    ax.plot(
        tm.r_range_fwd[-10:], dpsi_dr_fwd[-10:], label='np.gradient edge_order 2',
        linestyle='--'
    )
    
    dpsi_dr_bkwd = np.gradient(tm.psi_backwards, dr_bkwd, edge_order=2)
    ax2.plot(
        tm.r_range_bkwd[-10:], dpsi_dr_bkwd[-10:], linestyle='--'   
    )
    
    ax.legend()
    
    fig.supxlabel("Normalised minor radial co-ordinate (r/a)")
    fig.supylabel("Normalised perturbed flux gradient")
    
    fig.tight_layout()
    
    plt.savefig("gradient_test.png", dpi=300)
    
def q_sweep():
    modes = [
        (2,1),
        (3,2)
    ]

    axis_q_plt = np.linspace(0.0, 100.0, 200)

    lundquist_number = 1e8

    fig, ax = plt.subplots(1, figsize=(4.5,3))
    ax.hlines(
        0.0, min(axis_q_plt), max(axis_q_plt), color='black', linestyle='--',
        label='Marginal stability'
    )

    min_q, max_q = 1000000.0, -1000000.0

    for m,n in modes:
        print(m, n)
        q_rs = m/n
        axis_q = np.linspace(q_rs/q(0.0), q_rs/q(1.0), 50)
        results = [growth_rate(m,n,lundquist_number, q) for q in axis_q]
        delta_ps, growth_rates = zip(*results)

        min_q = min(min_q, min(axis_q))
        max_q = max(max_q, max(axis_q))

        ax.plot(axis_q, growth_rates, label=f"(m, n)=({m}, {n})")

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom=-top)
    ax.set_xlim(left=min_q, right=max_q)

    ax.set_xlabel("On-axis safety factor")
    ax.set_ylabel(r"Normalised growth rate ($\gamma/\bar{\omega}_A$)")

    ax.legend(prop={'size': 7}, ncol=2)
    fig.tight_layout()
    ax.grid(which='both')
    ms, ns = zip(*modes)
    savefig(
        f"q_vs_growth_rate_(m,n)_{min(ms)}-{max(ms)}_{min(ns)}-{max(ns)}"
    )
    plt.show()

if __name__=='__main__':
    solve_and_plot_system_simple()
    #plt.show()
    #plt.tight_layout()
    #plt.savefig("tm-with-q-djdr.png", dpi=300)

    # island_saturation()
    # plt.savefig("island-saturation.png", dpi=300)
    # plt.show()
    
    #print(growth_rate(4,2,1e8))
    growth_rate_vs_mode_number()
    #plt.show()

    #test_gradient()

    #q_sweep()
