import numpy as np

def ntm_bootstrap_term(w: float,
                       r_major: float,
                       f_val: float,
                       q_s: float,
                       shear_rs: float,
                       j_bs_avg_rs: float,
                       w_d: float = 0):
    """
    Calculate bootstrap current contribution to the MRE.

    See Kleiner NF 2016 equation 9.

    We normalise island width to minor radius. Delta' is
    normalised to minor radius also, i.e. we return a*Delta'.

    We also expand F=R.B_phi ~ R_0 * B_phi,0. This is valid
    in reduced MHD.

    Parameters
    ------------
    :param w: Magnetic island width to minor radius
    :param r_major: Major radius of the tokamak (metres)
    :param f_val: F=RBphi at rational surface
    :param q_s: Safety factor at rational surface
    :param shear_rs: Magnetic shear at rational surface
    :param j_bs_avg_rs: Poloidally averaged component of
        the bootstrap current in the direction of the
        magnetic field, i.e. <j_bs . Bvec>. Units of
        Tesla.Amp/m^2
    """
    mu0 = 4e-7 * np.pi
    return 64.0/(3*np.pi) * (
        mu0 * r_major**3 * q_s / (f_val**2 * shear_rs) *
        j_bs_avg_rs * w / (w**2 + w_d**2)
    )