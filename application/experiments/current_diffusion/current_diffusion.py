import numpy as np
from scipy.special import j0, j1, jn_zeros

def solve_diffusion(r: np.array,
                    t: float,
                    B_init: float,
                    B_applied: float,
                    n_harmonics: int = 10000) -> np.array:
    """
    Solve resistive diffusion of toroidal field in
    a cylinder assuming constant resistivity.

    Use r, theta, z co-ordinates.

    :param r: Radial co-ordinate normalised to minor radius a
    :param t: Time normalised to resistive timescale
    :param B_init: Initial uniform plasma toroidal field
    :param B_applied: Toroidal field applied to edge of plasma
    :param n_harmonics: Number of Bessel harmonics used in solution
    """
    zeros = jn_zeros(0, n_harmonics)
    j0_vals = j0(np.outer(zeros,r))
    j1_vals = j1(zeros)
    exp_decay = np.exp(-zeros**2 * t)

    c_n_vals = 2.0*exp_decay/(j1_vals*zeros)

    fourier_args = np.vecmat(c_n_vals, j0_vals)

    return B_init + (B_applied-B_init)*(1.0-fourier_args)

def q_profile_evolution(r: np.array,
                        t: float,
                        B_init: float,
                        B_applied: float,
                        nu: float = 1.0,
                        R_0: float = 1.0):
    """
    Model q-profile evolution from resistive diffusion
    of Bz-field onto axis.
    
    Assumes current profile of the form

    J_phi = J_0 * (1-r^2)^nu,

    where nu is a peaking factor.

    q profile can be derived analytically from this in a cylinder.
    See MSc diss, eq 2.5, 2.6.

    Use the constant resistivity solution

    :param r: Radial co-ordinate normalised to minor radius a
    :param t: Time normalised to resistive timescale
    :param B_init: Initial uniform plasma toroidal field
    :param B_applied: Toroidal field applied to edge of plasma
    :param n_harmonics: Number of Bessel harmonics used in solution
    :param nu: Shaping factor for current profile
    """
    B_z_prof = solve_diffusion(
        r, t, B_init, B_applied
    )

    q_prof = 2.0*B_z_prof/R_0 * (nu+1) * r**2 / (1-(1-r**2)**(nu+1))

    return q_prof

def get_rs_locations(q_prof: np.array,
                     r_vals: np.array,
                     q_res: float = 2.0) -> np.array:
    r_s_val = np.interp(q_res, q_prof, r_vals)
    return r_s_val


if __name__=='__main__':
    r = np.linspace(0.0, 0.999, 99)
    times = np.append([0.0], np.logspace(-3, 0, 10))
    B_init = 0.53
    B_applied = 0.9*B_init
    R0=1.0
    nu = 4.5
    # TODO: Determine scenario where q=2 surface can be moved
    # significantly without getting too close to q0=1
    # Look at effect of current peaking on efficiency of
    # moving the surface, i.e. how much does the surface move
    # and how quickly does it reach its new value

    from matplotlib import pyplot as plt
    import matplotlib as mpl

    fig, ax = plt.subplots(2, sharex=True, figsize=(6,4))
    ax_bphi, ax_q = ax
    for ax_i in ax:
        ax_i.grid()

    cmap = plt.get_cmap("viridis")

    r_s_vals = []
    for i,t in enumerate(times):
        color = cmap(i/len(times))

        sol = solve_diffusion(
            r, t, B_init, B_applied
        )

        ax_bphi.plot(
            r, sol, label=r"$t/\tau_r$="f"{t:.2g}",
            color=color
        )

        q_prof = q_profile_evolution(
            r, t, B_init, B_applied, nu, R_0=R0
        )
        r_s_vals.append(get_rs_locations(q_prof, r))

        ax_q.plot(r, q_prof, label=r"$t/\tau_r$="f"{t:.2g}", color=color)

    #ax_q.legend(ncol=2)
    ax_bphi.set_ylabel(r"$B_\phi(r)$")
    ax_q.set_ylabel(r"Safety factor")

    ax[-1].set_xlabel("$r/a$")

    cmap = plt.get_cmap("viridis", len(times))
    norm = mpl.colors.BoundaryNorm(times, cmap.N)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, ticks=times)
    cbar.set_label(r"$t/\tau_R$")

    #fig.tight_layout()

    fig2, ax2 = plt.subplots(1,figsize=(5,3))
    ax2.scatter(times, r_s_vals, color='black')
    ax2.set_xlabel(r"$t/\tau_R$")
    ax2.set_ylabel(r"$r(q=2)/a$")
    ax2.grid()
    fig2.tight_layout()

    plt.show()