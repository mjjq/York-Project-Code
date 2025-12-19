from argparse import ArgumentParser
import numpy as np
from scipy.interpolate import UnivariateSpline
from typing import Tuple
from dataclasses import dataclass

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

@dataclass
class XTORProfiles:
    n_psi: int
    r_mesh: np.array
    density_mesh: np.array
    t_ion_mesh: np.array
    t_electron_mesh: np.array
    ffprime_mesh: np.array

    # Below quantities are appended to 
    # ALL_PROFILES. Must be consistent with
    # XTOR
    aspct: float
    B0: float
    R0: float
    snumber: float
    slimit: float

    # These are typically left as default in XTOR
    norm_density: float = 1e20
    total_ni0_bulk: float = 1e19
    qi_bulk: float = 1.0
    mi_bulk: float = 2.0
    total_Ti0_bulk: float = 3.0
    total_Te0: float = 3.0

def generate_all_profiles_file(profiles: XTORProfiles, fname: str = 'ALL_PROFILES'):
    """
    Generate ALL_PROFILES file compatible with XTOR
    """
    xtor_lmax = profiles.n_psi+1
    n_p = 2*xtor_lmax + 1
    if not np.all([
        len(profiles.density_mesh)==n_p,
        len(profiles.t_electron_mesh)==n_p,
        len(profiles.t_ion_mesh)==n_p
    ]):
        raise ValueError("All meshes should have the same length!")

    with open(fname, 'w') as f:
        f.write(f"{2*xtor_lmax}  0  0\n")
        f.write("\n".join([f"{x}" for x in profiles.r_mesh]))
        f.write("\n")
        f.write("\n".join([f"{x}" for x in profiles.density_mesh]))
        f.write("\n")
        f.write("\n".join([f"{x}" for x in profiles.t_ion_mesh]))
        f.write("\n")
        f.write("\n".join([f"{x}" for x in profiles.t_electron_mesh]))
        f.write("\n")

        # Write equil quantities
        dummy_vars = [
            profiles.aspct,
            profiles.B0,
            profiles.R0,
            profiles.snumber,
            profiles.slimit,
            profiles.norm_density,
            profiles.total_ni0_bulk,
            profiles.qi_bulk,
            profiles.mi_bulk,
            profiles.total_Ti0_bulk,
            profiles.total_Te0
        ]
        f.write("\n".join([f"{v:.10e}" for v in dummy_vars]))

def generate_expeq_file(profiles: XTORProfiles, fname: str = 'EXPEQ_INIT'):
    """
    Generate EXPEQ_INIT file compatible with CHEASE

    For now, assume Z=0.
    """
    xtor_lmax = profiles.n_psi+1
    n_p = 2*xtor_lmax + 1
    if not np.all([
        len(profiles.density_mesh)==n_p,
        len(profiles.t_electron_mesh)==n_p,
        len(profiles.t_ion_mesh)==n_p
    ]):
        raise ValueError("All meshes should have the same length!")

    pressure_array = profiles.density_mesh*(profiles.t_electron_mesh + profiles.t_ion_mesh)

    pprime_spline = UnivariateSpline(
        profiles.r_mesh,
        pressure_array,
        s=0
    ).derivative()

    # Above spline gives dp/ds. We need dp/psi for CHEASE
    # which is dp/ds /(2.0*s) (ds/dpsi = 1/(2.0*s))
    pprime_array = pprime_spline(profiles.r_mesh)/(2.0*profiles.r_mesh)
    # Convert to CHEASE units
    pprime_array = pprime_array / profiles.aspct

    # Convert FF prime to CHEASE units
    ffprime_array = profiles.ffprime_mesh * profiles.aspct

    # Our pressure is given
    # in terms of XTOR units to convert to CHEASE units through
    p_ratio = 0.1*pressure_array[-1] * profiles.aspct**2

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(3)
    ax[0].plot(profiles.r_mesh, pressure_array)
    ax[1].plot(profiles.r_mesh, pprime_array)
    ax[2].plot(profiles.r_mesh, ffprime_array)
    plt.show()

    # See prof/save_files.f90. Following the format from that.
    with open(fname, 'w') as f:
        f.write(f"{profiles.aspct:.10e}\n")
        f.write(f"{0.0:.10e}\n") # Z=0.0
        f.write(f"{p_ratio:.10e}\n") # Edge pressure
        f.write(f"{2*xtor_lmax-2}\n") # Number of data points to write
        f.write(f"1\n") # nsttp=1 for reading FF' in CHEASE

        to_write = np.concatenate((
            profiles.r_mesh[2:-1], # Ignore array padding and r=0 index
            pprime_array[2:-1],
            ffprime_array[2:-1]    
        ))

        f.write("\n".join(f"{x:.10e}" for x in to_write))


def read_jorek_profile_ascii(filename: str) -> Tuple[np.array, np.array]:
    """
    Read JOREK ascii profile (two column format)

    JOREK profiles are given in terms of psi_N and the quantity
    Convert from psi_N to s via s = sqrt(psi_N)
    """
    data = np.loadtxt(filename)

    s = data[:,0]
    quantity = data[:,1]
    return s, quantity


def jorek_to_xtor_profiles(jorek_density_fname: str,
                           jorek_temp_fname: str,
                           jorek_ffprime_fname: str,
                           epsilon: float,
                           B0: float,
                           npsi: int,
                           R0: float = 0.0,
                           snumber: float = 0.0,
                           slimit: float = 0.0) -> XTORProfiles:
    """
    Convert JOREK density and temperature profiles to
    a format accepted by XTOR (via ALL_PROFILES)
    """
    # Definition of lmax as per XTOR docs
    xtor_lmax = npsi+1
    delta_r = 1.0/npsi
    density_rs, density_vals = read_jorek_profile_ascii(jorek_density_fname)
    temp_rs, temp_vals = read_jorek_profile_ascii(jorek_temp_fname)
    ff_rs, ff_vals = read_jorek_profile_ascii(jorek_ffprime_fname)

    # d_r_filter = density_rs < 1.0+delta_r/2.0
    # t_r_filter = temp_rs < 1.0+delta_r/2.0

    # density_rs = density_rs[d_r_filter]
    # density_vals = density_vals[d_r_filter]
    # temp_rs = temp_rs[t_r_filter]
    # temp_vals = temp_vals[t_r_filter]

    # Convert from JOREK units to XTOR units
    temp_vals = temp_vals / (B0*epsilon)**2
    ff_vals = ff_vals / (epsilon*B0)

    # Rescale from psi_N to s=sqrt(psi_N)
    density_rs_rescale = upscale_profile(density_rs, 2*xtor_lmax)**0.5
    density_val_rescale = upscale_profile(density_vals, 2*xtor_lmax)
    temp_rs_rescale = upscale_profile(temp_rs, 2*xtor_lmax)**0.5
    temp_val_rescale = upscale_profile(temp_vals, 2*xtor_lmax)
    ff_rs_rescale = upscale_profile(ff_rs, 2*xtor_lmax)**0.5
    ff_val_rescale = upscale_profile(ff_vals, 2*xtor_lmax)

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
    ff_val_rescale = sample_radial_profile(
        ff_rs_rescale,
        ff_val_rescale,
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

    left_pad_ff = [ff_val_rescale[0]]
    ff_val_rescale = np.concatenate((left_pad_ff, ff_val_rescale))


    #from matplotlib import pyplot as plt

    #plt.plot(density_rs_rescale, density_val_rescale)
    #plt.scatter(temp_rs**0.5, temp_vals)
    #plt.scatter(linear_rs_profile, temp_val_rescale)
    #plt.scatter(density_rs**0.5, density_vals)
    #plt.scatter(linear_rs_profile, density_val_rescale)
    #plt.show()

    return XTORProfiles(
        npsi,
        linear_rs_profile,
        density_val_rescale,
        0.5*temp_val_rescale,
        0.5*temp_val_rescale,
        ff_val_rescale,
        epsilon,
        B0,
        R0,
        snumber,
        slimit
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
    parser.add_argument('jorek_ffprime', type=str, help="Path to JOREK ffprime file")
    parser.add_argument('-n','--npsi-chease', type=int, help="Number of flux surface (==NPSI in chease)")

    parser.add_argument('-a','--aspct',type=float, help="Inverse aspect ratio of plasma (a/R_0)")
    parser.add_argument('-b','--B0', type=float, help="On-axis toroidal field of plasma (T)")
    parser.add_argument('-r','--R0',type=float, help="Major radius of plasma (m)")
    parser.add_argument('-s','--snumber', type=float, help="On-axis lundquist number")
    parser.add_argument('-sl','--slimit', type=float, help="Lundquist number limit")


    args = parser.parse_args()

    profiles = jorek_to_xtor_profiles(
        args.jorek_density,
        args.jorek_temperature,
        args.jorek_ffprime,
        args.aspct,
        args.B0,
        args.npsi_chease,
        args.R0,
        args.snumber,
        args.slimit
    )

    generate_all_profiles_file(profiles)
    generate_expeq_file(profiles)
