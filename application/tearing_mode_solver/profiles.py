import numpy as np
from scipy.optimize import minimize_scalar

def dj_dr(radial_coordinate: float,
          shaping_exponent: float) -> float:
    """
    Normalised derivative in the current profile.

    Current is normalised to the on-axis current J_0.
    radial_coordinate is normalised to the minor radius a.
    """
    r = radial_coordinate
    nu = shaping_exponent

    return -2*nu*(r) * (1-r**2)**(nu-1)

@np.vectorize
def q(radial_coordinate: float,
      shaping_exponent: float) -> float:
    r = radial_coordinate
    nu = shaping_exponent

    # Prevent division by zero for small r values.
    # For this function, in the limit r->0, q(r)->1. This is proven
    # mathematically in the lab book.
    #print(r)
    if np.abs(r) < 1e-5:
        return 1.0

    return (nu+1)*(r**2)/(1-(1-r**2)**(nu+1))

def rational_surface(target_q: float,
                     shaping_exponent: float) -> float:
    """
    Compute the location of the rational surface of the q-profile defined in q().
    """

    # Establish the function to pass to scipy's scalar minimiser. We want to
    # find the value of r such that q(r) = target_q. This is equivalent to
    # finding the value of r that minimises (q(r) - target_q)^2
    # Call vectorize so that it converts q(r) into a function that can take
    # np arrays. This is necessary because q(r) contains an if statement which
    # fails if the function is not vectorized.
    fun = np.vectorize(lambda r : (q(r, shaping_exponent) - target_q)**2)

    rs = minimize_scalar(fun, bounds=(0.0, 1.0), method='bounded')

    return rs.x
