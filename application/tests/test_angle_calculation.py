import numpy as np
import numpy.testing as npt
import os
import pytest
from typing import Tuple

from jorek_tools.poincare.q_profile_poincare import angle_between_two_vectors, Vec2D


@pytest.mark.parametrize(
    "test, expected",
    [
        ( ((0.0, 1.0), (0.0, 1.0)), 0.0),
        ( ((0.0, 1.0), (0.0, -1.0)), np.pi),
        ( ((0.0, 1.0), (-1.0, 0.0)), np.pi/2.0),
        ( ((0.0, 1.0), (1.0, 0.0)), 3.0*np.pi/2.0),
        ( ((0.0, 1.0), (-1.0/np.sqrt(2), 1.0/np.sqrt(2))), np.pi/4),
        ( ((0.0, 1.0), (-1.0/np.sqrt(2), -1.0/np.sqrt(2))), 3*np.pi/4),
        ( ((0.0, 1.0), (1.0/np.sqrt(2), -1.0/np.sqrt(2))), 5*np.pi/4),
        ( ((0.0, 1.0), (1.0/np.sqrt(2), 1.0/np.sqrt(2))), 7*np.pi/4),
        ( ((-1.0, 0.0), (-1.0/np.sqrt(2), -1.0/np.sqrt(2))), np.pi/4),
        ( ((-1.0, 0.0), (-1.0/np.sqrt(2), 1.0/np.sqrt(2))), 7*np.pi/4)
    ])
def test_angles(test: Tuple[Vec2D, Vec2D], expected: float):
    vec1, vec2 = test
    angle = 360.0*angle_between_two_vectors(vec1, vec2)/(2.0*np.pi)
    expected_deg = 360.0*expected/(2.0*np.pi)

    npt.assert_approx_equal(angle, expected_deg)