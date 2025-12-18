from argparse import ArgumentParser
import numpy as np
from scipy.interpolate import UnivariateSpline
from typing import Tuple
import f90nml

def upscale_profile(prof: np.array,
                    new_length: int) -> np.array:
    """
    Upscale an array from its current size to a new size

    :param r_prof: r-mesh
    :param q_prof: Quantity profile
    """
    space = np.linspace(0.0, 1.0, len(prof))

    spline = UnivariateSpline(space, prof, s=0.0)

    new_points = np.linspace(0.0, 1.0, new_length)

    return spline(new_points)

def sample_radial_profile(prof_r: np.array,
                          prof_val: np.array,
                          new_prof_r: np.array) -> np.array:
    spline = UnivariateSpline(prof_r, prof_val, s=0)

    return spline(new_prof_r)

def generate_all_profiles_file(n_psi: int,
                               r_mesh: np.array,
                               density_mesh: np.array,
                               t_ion_mesh: np.array,
                               t_electron_mesh: np.array):
    """
    Generate ALL_PROFILES file compatible with XTOR

    :param r_mesh: List describing radial points
    :param density_mesh: List describing density at each radial point
    :param t_electron_mesh: List describing electron temperature at
        each radial point
    :param t_ion_mesh: List describing ion temperature at
        each radial point
    """
    xtor_lmax = n_psi+1
    n_p = 2*xtor_lmax + 1
    if not np.all([
        len(density_mesh)==n_p,
        len(t_electron_mesh)==n_p,
        len(t_ion_mesh)==n_p
    ]):
        raise ValueError("All meshes should have the same length!")

    with open('ALL_PROFILES', 'w') as f:
        f.write(f"{2*xtor_lmax}  0  0\n")
        f.write("\n".join([f"{x}" for x in r_mesh]))
        f.write("\n")
        f.write("\n".join([f"{x}" for x in density_mesh]))
        f.write("\n")
        f.write("\n".join([f"{x}" for x in t_ion_mesh]))
        f.write("\n")
        f.write("\n".join([f"{x}" for x in t_electron_mesh]))
        f.write("\n")

        # Write dummy equil quantities
        f.write("\n".join(["0" for x in range(11)]))

def read_jorek_profile_ascii(filename: str) -> Tuple[np.array, np.array]:
    """
    Read JOREK ascii profile (two column format)

    JOREK profiles are given in terms of psi_N and the quantity
    of interest.

    Convert from psi_N to s via s = sqrt(psi_N)
    """
    data = np.loadtxt(filename)

    s = data[:,0]
    quantity = data[:,1]
    return s, quantity


def jorek_to_xtor_profiles(jorek_density_fname: str,
                           jorek_temp_fname: str,
                           epsilon: float,
                           B0: float,
                           npsi: int):
    """
    Convert JOREK density and temperature profiles to
    a format accepted by XTOR (via ALL_PROFILES)

    :param jorek_density_fname: Path to JOREK density filename
    :param jorek_temp_fname: Path to JOREK temperature filename
    :param epsilon: Aspect ratio of the system
    :param B0: On-axis toroidal field of the system
    :param xtor_lmax: Number of radial points in XTOR
    """
    # Definition of lmax as per XTOR docs
    xtor_lmax = npsi+1
    delta_r = 1.0/npsi
    density_rs, density_vals = read_jorek_profile_ascii(jorek_density_fname)
    temp_rs, temp_vals = read_jorek_profile_ascii(jorek_temp_fname)

    # d_r_filter = density_rs < 1.0+delta_r/2.0
    # t_r_filter = temp_rs < 1.0+delta_r/2.0

    # density_rs = density_rs[d_r_filter]
    # density_vals = density_vals[d_r_filter]
    # temp_rs = temp_rs[t_r_filter]
    # temp_vals = temp_vals[t_r_filter]

    # Convert from JOREK units to XTOR units
    temp_vals = temp_vals * (B0*epsilon)**2

    density_rs_rescale = upscale_profile(density_rs, 2*xtor_lmax)**0.5
    density_val_rescale = upscale_profile(density_vals, 2*xtor_lmax)
    temp_rs_rescale = upscale_profile(temp_rs, 2*xtor_lmax)**0.5
    temp_val_rescale = upscale_profile(temp_vals, 2*xtor_lmax)

    # In prof, r(xtor_lmax)=1+delta_r/2 by definition
    linear_rs_profile = np.linspace(0.0, 1.0+delta_r/2.0, 2*xtor_lmax)

    temp_val_rescale = sample_radial_profile(
        temp_rs_rescale,
        temp_val_rescale,
        linear_rs_profile
    )
    density_val_rescale = sample_radial_profile(
        density_rs_rescale,
        density_val_rescale,
        linear_rs_profile
    )

    # Now add left padding to arrays. Right padding already
    # included
    left_pad_r = [-delta_r/2.0]
    linear_rs_profile = np.concatenate((left_pad_r, linear_rs_profile))

    left_pad_n = [density_val_rescale[0]]
    density_val_rescale = np.concatenate((left_pad_n, density_val_rescale))

    left_pad_t = [temp_val_rescale[0]]
    temp_val_rescale = np.concatenate((left_pad_t, temp_val_rescale))


    #from matplotlib import pyplot as plt

    #plt.plot(density_rs_rescale, density_val_rescale)
    #plt.scatter(temp_rs**0.5, temp_vals)
    #plt.scatter(linear_rs_profile, temp_val_rescale)
    #plt.scatter(density_rs**0.5, density_vals)
    #plt.scatter(linear_rs_profile, density_val_rescale)
    #plt.show()

    generate_all_profiles_file(
        npsi,
        linear_rs_profile,
        density_val_rescale,
        0.5*temp_val_rescale,
        0.5*temp_val_rescale
    )  


def _compare_upscaled_profile():
    r_old = np.linspace(-5.0, 5.0, 20)
    q_old = np.tanh(r_old)

    r_rescale = upscale_profile(r_old, 500)
    q_rescale = upscale_profile(q_old, 500)

    from matplotlib import pyplot as plt

    plt.plot(r_old, q_old)
    plt.plot(r_rescale, q_rescale)
    
    plt.show()

def _test_all_profs_gen():
    r_mesh = np.linspace(0.0, 1.0, 244)
    n_mesh = np.linspace(0.5, 0.5, 244)
    ti_mesh = np.tanh(r_mesh)
    te_mesh = np.tanh(r_mesh)

    generate_all_profiles_file(r_mesh, n_mesh, ti_mesh, te_mesh)


if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('jorek_density', type=str, help="Path to jorek density file")
    parser.add_argument('jorek_temperature', type=str, help="Path to JOREK temperature file")
    parser.add_argument('-n','--npsi-chease', type=int, help="Number of flux surface (==NPSI in chease)")
    parser.add_argument('-b','--B0', type=float, help="On-axis toroidal field of plasma")
    parser.add_argument('-e','--epsilon',type=float, help="Inverse aspect ratio of plasma (a/R_0)")

    args = parser.parse_args()

    jorek_to_xtor_profiles(
        args.jorek_density,
        args.jorek_temperature,
        args.epsilon,
        args.B0,
        args.npsi_chease
    )
