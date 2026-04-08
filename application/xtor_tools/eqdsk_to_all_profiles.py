from freeqdsk import geqdsk
import numpy as np

def standard_density_profile(psin_vals: np.array):
    """
    Standard mtanh dropoff in density
    """
    return 

def eqdsk_to_xtor_profiles(eqdsk_filename: str,
                           npsi: int,
                           psin_limit: float = 0.95,
                           snumber: float = 0.0,
                           slimit: float = 0.0) -> XTORProfiles:
    """
    Convert EQDSK profiles to
    a format accepted by XTOR (via ALL_PROFILES)
    """
    # Definition of lmax as per XTOR docs
    xtor_lmax = npsi+1
    delta_r = 1.0/npsi
    density_rs, density_vals = read_jorek_profile_ascii(jorek_density_fname)
    temp_rs, temp_vals = read_jorek_profile_ascii(jorek_temp_fname)
    ff_rs, ff_vals = read_jorek_profile_ascii(jorek_ffprime_fname)

    print(density_rs[0], density_vals[0])
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
    density_psi_vals = upscale_profile(density_rs, 2*xtor_lmax)
    density_rs_rescale = density_psi_vals**0.5
    density_val_rescale = upscale_profile(density_vals, 2*xtor_lmax)

    temp_psi_vals = upscale_profile(temp_rs, 2*xtor_lmax)
    temp_rs_rescale = temp_psi_vals**0.5
    temp_val_rescale = upscale_profile(temp_vals, 2*xtor_lmax)
    ff_rs_rescale = upscale_profile(ff_rs, 2*xtor_lmax)**0.5
    ff_val_rescale = upscale_profile(ff_vals, 2*xtor_lmax)

    # Get pressure as function of psi
    psi_array = np.linspace(0.0, 1.0, 2*xtor_lmax)
    # Resample all arrays to ensure they have same radial scale
    density_psi = sample_radial_profile(
        density_psi_vals,
        density_val_rescale,
        psi_array
    )
    temp_psi = sample_radial_profile(
        temp_psi_vals,
        temp_val_rescale,
        psi_array
    )
    pressure_psi = density_psi * temp_psi
    p_psi_spline = UnivariateSpline(
        psi_array,
        pressure_psi,
        s=0
    )

    dp_dpsi_spline = p_psi_spline.derivative()
    dp_dpsi = dp_dpsi_spline(psi_array)

    from matplotlib import pyplot as plt
    plt.plot(psi_array, dp_dpsi)

    # In prof, r(xtor_lmax)=1+delta_r/2 by definition
    linear_rs_profile = np.linspace(0.0, 1.0+delta_r/2.0, 2*xtor_lmax)

    # Now resample to get dp_dpsi in terms of s
    dp_dpsi = sample_radial_profile(
        psi_array**0.5,
        dp_dpsi,
        linear_rs_profile
    )

    plt.plot(linear_rs_profile, dp_dpsi)


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

    left_pad_pp = [dp_dpsi[0]]
    pp_val_rescale = np.concatenate((left_pad_pp, dp_dpsi))


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
        pp_val_rescale,
        epsilon,
        B0,
        R0,
        snumber,
        slimit
    ) 

if __name__=='__main__':
    import sys
    from matplotlib import pyplot as plt

    filename = sys.argv[1]

    with open(filename, "r") as fh:
        gfile = geqdsk.read(fh)

        psin = np.linspace(0.0, 1.0, gfile.nx)

        plt.plot(psin, gfile.pres)

        plt.show()