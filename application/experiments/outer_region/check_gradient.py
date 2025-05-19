from matplotlib import pyplot as plt

import imports
from tearing_mode_solver.outer_region_solver import *



def test_gradient():
    """
    Compare the derivative of the outer solution calculated by ODEINT against
    that calculated manually from the perturbed flux using numpy.gradient.
    """
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

    plt.show()

if __name__=='__main__':
    test_gradient()
