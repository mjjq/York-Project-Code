import pytest
from numpy.testing import assert_almost_equal

@pytest.mark.parametrize(
        "chi_perp,chi_parallel,expected_wd", 
        [(1e-8, 1.0, 0.0486), (1e-9, 1.0, 0.0274)]
    )
def test_diffusion_width(chi_perp, chi_parallel, expected_wd):
    """
    Test diffusion width according to Lutjens, PoP, 2001.
    """
    from tearing_mode_solver.outer_region_solver import diffusion_width

    # Other parameters used in Lutjens 2001
    r_s = 0.58
    mag_shear = 0.445
    aspect_ratio = 2.56
    toroidal_mode_number = 1

    diff_width = diffusion_width(
        chi_perp,
        chi_parallel,
        r_s,
        aspect_ratio,
        toroidal_mode_number,
        mag_shear
    )

    # Values okay if within 10 percent of eachother
    percentage_diff = 100.0 * abs((diff_width-expected_wd)/expected_wd)
    
    assert percentage_diff < 10.0



def test_diff_width_inverse():
    """
    Test inverse of diffusion width actually works
    """
    import numpy as np
    from tearing_mode_solver.outer_region_solver import (
        diffusion_width,
        chi_perp_ratio
    )

    chi_perps = np.linspace(0.0, 1.0, 10000)
    chi_parallels = np.ones(10000)
    chi_ratios = chi_perps/chi_parallels

    # Other parameters used in Lutjens 2001
    r_s = 0.58
    mag_shear = 0.445
    aspect_ratio = 2.56
    toroidal_mode_number = 1

    diff_widths = diffusion_width(
        chi_perps,
        chi_parallels,
        r_s,
        aspect_ratio,
        toroidal_mode_number,
        mag_shear
    )

    derived_chi_ratios = chi_perp_ratio(
        diff_widths,
        r_s,
        aspect_ratio,
        toroidal_mode_number,
        mag_shear
    )

    np.testing.assert_array_almost_equal(
        chi_ratios, derived_chi_ratios
    )