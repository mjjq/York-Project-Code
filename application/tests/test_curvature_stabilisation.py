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

    assert_almost_equal(diff_width, expected_wd, decimal=4)

