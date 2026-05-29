import numpy as np

from experiments.ntm_modelling.mre_time_series import MREContributions

def compare_dw_dt(mre: MREContributions):
    dw_dt_measured = np.diff(mre.w_measured)/np.diff(mre.times)

    delta_prime_total = (
        mre.delta_p_cl_finite_island + mre.delta_p_bs + mre.delta_p_ggj
    )

    mu0 = 4e-7 * np.pi

    dw_dt_theory = mre.resistivity/mu0 * delta_prime_total

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1)

    ax.plot(mre.times[:-1], dw_dt_measured, label='Measured (JOREK)')
    ax.plot(mre.times, dw_dt_theory, label='MRE')

    plt.show()
