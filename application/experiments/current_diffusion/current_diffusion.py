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

def solve_diffusion_tdep(r: np.array,
                    t: float,
                    B_init: float,
                    B_applied: float,
                    tau_coil: float,
                    n_harmonics: int = 10000) -> np.array:
    """
    Solve resistive diffusion of toroidal field in
    a cylinder assuming constant resistivity and a linear
    rampdown of the magnetic field.

    I.e., B(t)=B_init + (B_applied-B_init)*t/tau_coil for 0<t<tau_coil
          B(t) = B_applied for t >= tau_coil

    Use r, theta, z co-ordinates.

    :param r: Radial co-ordinate normalised to minor radius a
    :param t: Time normalised to resistive timescale
    :param B_init: Initial uniform plasma toroidal field
    :param B_applied: Toroidal field applied to edge of plasma
    :param tau_coil: Ramp down time of the toroidal coil in units
        of resistive time
    :param n_harmonics: Number of Bessel harmonics used in solution
    """
    zeros = jn_zeros(0, n_harmonics)
    j0_vals = j0(np.outer(zeros,r))
    j1_vals = j1(zeros)
    t_arg = min(t, tau_coil)
    exp_decay = np.exp(-zeros**2 * t)*(np.exp(-zeros**2 * t_arg)-1.0)/(zeros**2 * tau_coil)

    c_n_vals = 2.0*exp_decay/(j1_vals*zeros)

    fourier_args = np.vecmat(c_n_vals, j0_vals)

    if t < tau_coil:
        return B_init + (B_applied-B_init)*(t/tau_coil + fourier_args)
    else:
        return B_applied + (B_applied-B_init)*fourier_args

def q_profile(r: np.array,
                B_z: np.array,
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
    :param B_z: 2D array of toroidal field values (in r,t space)
    :param nu: Shaping factor for current profile
    """
    q_prof = 2.0*B_z/R_0 * (nu+1) * r**2 / (1-(1-r**2)**(nu+1))

    return q_prof

def get_rs_locations(q_prof: np.array,
                     r_vals: np.array,
                     q_res: float = 2.0) -> np.array:
    r_s_val = np.interp(q_res, q_prof, r_vals)
    return r_s_val


if __name__=='__main__':
    r = np.linspace(0.0, 0.999, 99)
    times = np.append([0.0], np.logspace(-3, 1, 100))
    tau_coils = [0.01, 1.0, 10.00]
    B_init = 0.53
    B_applied = 0.9*B_init
    R0=1.0
    nu = 4.5
    # TODO: Determine scenario where q=2 surface can be moved
    # significantly without getting too close to q0=1
    # Look at effect of current peaking on efficiency of
    # moving the surface, i.e. how much does the surface move
    # and how quickly does it reach its new value
    # TODO: Significant modification to shear at edge, could this be
    # used for peeling mode mitigation?

    from matplotlib import pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as tkr

    # All on same plot
    # fig, axs = plt.subplots(3,3,figsize=(8,6))
    # for i,row in enumerate(axs):
    #     for j, subplot in enumerate(row):
    #         subplot.sharey(row[0])
    #         subplot.sharex(axs[0,j])
    #         subplot.label_outer()
    #figs = [fig]
    # for ax_i in axs[-1]:
    #     ax_i.set_xlabel("$r/a$")

    fig1, ax1 = plt.subplots(1,3, sharex=True, sharey=True, figsize=(9,3))
    fig2, ax2 = plt.subplots(1,3, sharex=True, sharey=True, figsize=(9,3))
    fig3, ax3 = plt.subplots(1,3, sharex=True, sharey=True, figsize=(9,3))
    axs=np.array([ax1, ax2, ax3])
    figs = [fig1, fig2, fig3]
    for ax in axs.flatten():
        ax.label_outer()
        ax.set_xlabel("$r/a$")
            

    r_s_vals_tb = []
    r_s_q1_vals_tb = []
    shear_vals_tb = []
    for i, tau_coil in enumerate(tau_coils):
        ax = axs[:,i]
        ax_bphi, ax_q, ax_s = ax
        for ax_i in ax:
            ax_i.grid()
        if i==0:
            ax_bphi.set_ylabel(r"$B_\phi(r)$ [T]")
            ax_q.set_ylabel(r"Safety factor")
            ax_s.set_ylabel(r"Mag. shear")
        ax_bphi.set_title(r"$\tau_b/\tau_r=$"f"{tau_coil}")
        ax_q.set_title(r"$\tau_b/\tau_r=$"f"{tau_coil}")
        ax_s.set_title(r"$\tau_b/\tau_r=$"f"{tau_coil}")

        cmap = plt.get_cmap("viridis")

        r_s_vals = []
        r_s_q1_vals = []
        shear_vals = []
        for i,t in enumerate(times):
            color = cmap(i/len(times))

            sol = solve_diffusion_tdep(
                r, t, B_init, B_applied, tau_coil
            )

            ax_bphi.plot(
                r, sol, label=r"$t/\tau_r$="f"{t:.3g}",
                color=color
            )

            q_prof = q_profile(
                r, sol, nu, R_0=R0
            )
            dq_dr = np.diff(q_prof)/np.diff(r)
            shear = (r/q_prof)[:-1]*dq_dr
            r_s = get_rs_locations(q_prof, r)
            r_s_vals.append(r_s)
            shear_vals.append(np.interp(r_s, r[:-1], shear))

            ax_q.plot(r, q_prof, label=r"$t/\tau_r$="f"{t:.3g}", color=color)
            ax_s.plot(r[:-1], shear,color=color)

        r_s_vals_tb.append(r_s_vals)
        shear_vals_tb.append(shear_vals)

    #ax_q.legend(ncol=2)


    cmap = plt.get_cmap("viridis", len(times))
    norm = mpl.colors.BoundaryNorm(times, cmap.N)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i,fig in enumerate(figs):
        ax_fig = fig.get_axes()
        cbar = fig.colorbar(sm, ax=ax_fig[-1], ticks=times, format=tkr.FormatStrFormatter('%.3f'))
        cbar.set_label('%.2g')
        cbar.set_label(r"$t/\tau_r$")

        fig.tight_layout()
        fig.savefig(f"current_diffusion_{i}.pdf")

    fig2, ax2 = plt.subplots(1,figsize=(5,3))
    ax2.set_prop_cycle(linestyle=['-','--',':'], color=['black', 'red', 'blue'])
    for t_coil,r_s_vals in zip(tau_coils,r_s_vals_tb):
        ax2.plot(times, r_s_vals, label=r"$\tau_b/\tau_r=$"f"{t_coil:.2f}")
    ax2.set_xlabel(r"$t/\tau_r$")
    ax2.set_ylabel(r"$r(q=2)/a$")
    ax2.grid()
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("current_diffusion_rs.pdf")
    ax2.set_xscale('log')
    fig2.tight_layout()
    fig2.savefig("current_diffusion_rs_log.pdf")

    fig3, ax3 = plt.subplots(1, figsize=(5,3))
    ax3.set_prop_cycle(linestyle=['-','--',':'], color=['black', 'red', 'blue'])
    for t_coil, shear_vals in zip(tau_coils, shear_vals_tb):
        ax3.plot(times, shear_vals, label=r"$\tau_b/\tau_r=$"f"{t_coil:.2f}")
    ax3.set_xlabel(r"$t/\tau_r$")
    ax3.set_ylabel(r"Mag. shear $(q=2)$")
    ax3.grid()
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig("current_diffusion_shear.pdf")
    ax3.set_xscale('log')
    fig3.tight_layout()
    fig3.savefig("current_diffusion_shear_log.pdf")

    plt.show()