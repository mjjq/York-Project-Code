from argparse import ArgumentParser
import numpy as np

from jorek_tools.macroscopic_vars_analysis.plot_quantities import (
    MacroscopicQuantity, plot_macroscopic_quantities
)
from tearing_mode_solver.helpers import TimeDependentSolution

def jorek_energy_to_dpsi(jorek_energy_data: MacroscopicQuantity,
                         delta_psi_0=1e-12) -> TimeDependentSolution:
    """
    Take loaded magnetic energy data from a JOREK run, convert it to delta_psi
    normalised to delta_psi_0. delta_psi_0 marks the "t=0" point of the simulation,
    i.e. all times are shifted such that t=0 corresponds to delta_psi_0.

    Note: This assumes that there is only one dominant poloidal mode in the plasma,
    i.e. a 2/1 tearing mode. 

    :param jorek_energy_data: Data loaded from magnetic_energies.dat generated
    from JOREK
    :param delta_psi_0: Reference "beginning" point of the simulation. 
    """
    # Load times on the x-axis (index 0)
    jorek_energy_data.load_x_values_by_index(0)
    # Load n=1 energies on y-axis (index 2)
    jorek_energy_data.load_y_values_by_index(2)

    times = jorek_energy_data.x_values
    # Convert from energy to some value propto \delta\psi
    # by taking square root
    delta_psi = jorek_energy_data.y_values**0.5 / delta_psi_0

    t0 = np.interp(
        1.0, delta_psi, times
    )

    times = times - t0

    filt = times > 0.0
    times = times[filt]
    delta_psi = delta_psi[filt]
    dpsi_dt = np.diff(delta_psi)/np.diff(times)
    d2psi_dt2 = np.diff(dpsi_dt)/np.diff(times)[:-1]

    return TimeDependentSolution(
        times,
        delta_psi,
        dpsi_dt,
        d2psi_dt2,
        None,
        None
    )




if __name__=='__main__':
    parser = ArgumentParser(
        description = "Plot tearing mode energies (or delta psi) scaled to resistivity"
    )

    parser.add_argument(
        "filenames",
        help="List of files containing mode energies",
        nargs='+',
        type=str
    )
    parser.add_argument(
        "-al", '--align-at',
        help="Value of delta psi at which all modes are aligned",
        type=float,
        default=1e-12
    )

    args = parser.parse_args()

    filenames = args.filenames

    mqs = []
    labels = []
    etas = np.logspace(-6, -10, 5)

    for i,filename in enumerate(filenames):
        mq = MacroscopicQuantity(filename)
        mq.load_x_values_by_index(0)
        mq.load_y_values_by_index(2)

        # Convert from energy to some value propto \delta\psi
        # by taking square root
        mq.y_values = mq.y_values**0.5


        #mq.x_values = mq.x_values*(etas[i]/etas[0])**(3/5)


        aligned_dpsi_time = np.interp(
            args.align_at, mq.y_values, mq.x_values
        )
        mq.x_values = mq.x_values - aligned_dpsi_time

        delta_psi_1 = 10.0*args.align_at
        t1 = np.interp(delta_psi_1, mq.y_values, mq.x_values)

        dpsi_dt = (delta_psi_1 - args.align_at)/(t1-0.0)
        growth_rate = dpsi_dt/delta_psi_1

        mq.x_values = mq.x_values*growth_rate

        filt = mq.x_values > 0.0
        mq.x_values = mq.x_values[filt]
        mq.y_values = mq.y_values[filt]

        mqs.append(mq)
        labels.append(r"$\eta="f"{etas[i]:.2g}$")

    
    plot_macroscopic_quantities(
        mqs,
        labels,
        r"Time [$\gamma \cdot$ JOREK UNITS]",
        r"$\delta\psi$ (arb)",
        'linear', 'log',
        (4,3),
        None,
        None,
        '-',
        1.0,
        None
    )
