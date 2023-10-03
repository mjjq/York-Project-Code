import numpy as np

def dj_dr(radial_coordinate: float,
          shaping_exponent: float = 2.0) -> float:
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
      shaping_exponent: float = 2.0) -> float:
    r = radial_coordinate
    nu = shaping_exponent

    # Prevent division by zero for small r values.
    # For this function, in the limit r->0, q(r)->1. This is proven
    # mathematically in the lab book.
    #print(r)
    if np.abs(r) < 1e-5:
        return 1.0

    return (nu+1)*(r**2)/(1-(1-r**2)**(nu+1))
