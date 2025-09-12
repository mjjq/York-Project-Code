from argparse import ArgumentParser
import numpy as np

from jorek_tools.macroscopic_vars_analysis.plot_quantities import (
    MacroscopicQuantity, plot_macroscopic_quantities
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
