import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from xtor_tools.get_equil_profiles import get_equil_rmaj_profiles



if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'jorek_filename', type=str, help='Path to JOREK rmajor profiles (postproc output)'
    )
    parser.add_argument(
        'xtor_filename', type=str, help='Path to XTOR xtfields filename'
    )

    args = parser.parse_args()

    xtor_profs = get_equil_rmaj_profiles(args.xtor_filename)
    jorek_profs = np.loadtxt(args.jorek_filename, comments='#')
    print(jorek_profs)

    jorek_profs = jorek_profs[jorek_profs[:,0]>0.0]

    jorek_rmaj = jorek_profs[:,0]
    jorek_p_e = jorek_profs[:,9]
    jorek_p_i = jorek_profs[:,10]
    jorek_jphi = jorek_profs[:,3]

    jorek_rnorm = (jorek_rmaj - 2.5)/0.5
    jorek_jphi = -2.5*jorek_jphi
    jorek_p_e_x = jorek_p_e / 0.2**2
    jorek_p_i_x = jorek_p_i / 0.2**2

    xtor_rnorm = (xtor_profs.r_maj-5.0)

    xtor_jphi = xtor_profs.j_phi[xtor_profs.j_phi>0.0]
    xtor_rnorm_jphi = xtor_rnorm[xtor_profs.j_phi>0.0]

    fig, axs = plt.subplots(4, figsize=(5,6), sharex=True)
    ax_pi, ax_pe, ax_p, ax_jphi = axs
    for ax in axs:
        ax.grid()

    ax_pi.plot(jorek_rnorm, jorek_p_i_x, label='JOREK')
    ax_pi.plot(xtor_rnorm, xtor_profs.p_i, label='XTOR')
    ax_pi.set_ylabel("$P_i$ [XU]")
    ax_pi.legend()

    ax_pe.plot(jorek_rnorm, jorek_p_e_x, label='JOREK')
    ax_pe.plot(xtor_rnorm, xtor_profs.p_e, label='XTOR')
    ax_pe.set_ylabel("$P_e$ [XU]")

    ax_p.plot(jorek_rnorm, jorek_p_e_x+jorek_p_i_x, label='JOREK')
    ax_p.plot(xtor_rnorm, xtor_profs.p_i+xtor_profs.p_e, label='XTOR')
    ax_p.set_ylabel("$P_e+P_i$ [XU]")
    
    ax_jphi.plot(jorek_rnorm, jorek_jphi, label='JOREK')
    ax_jphi.plot(xtor_rnorm_jphi, xtor_jphi, label='XTOR')
    ax_jphi.set_ylabel(r"$\langle J_\phi \rangle$ [XU]")

    axs[-1].set_xlabel(r'$(R-R_0)/a$')

    fig.tight_layout()

    plt.show()


