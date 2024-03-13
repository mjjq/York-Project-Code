import numpy as np
from scipy.optimize import minimize_scalar
from typing import List, Tuple
from scipy.interpolate import UnivariateSpline

@np.vectorize
def j(radial_coordinate: float,
      shaping_exponent: float) -> float:
    """
    Current profile function
    
    Current normalised to the on-axis current J_0.
    Radial co-ordinate normalised to the minor radius a.

    """
    r = radial_coordinate
    nu = shaping_exponent
    
    return (1-r**2)**nu
    
    
@np.vectorize
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

def generate_j_profile(shaping_exponent: float) -> List[Tuple[float, float]]:
    """
    Generate a current profile from the j() function defined above

    Parameters
    ----------
    shaping_exponent : float
        DESCRIPTION.

    Returns
    -------
    List[Tuple[float, float]]
        List containing elements of the form (minor_radial_coord, j_at_coord).

    """
    r_values = np.linspace(0.0, 1.0, 100)
    j_values = j(r_values, shaping_exponent)
    
    return list(zip(r_values, j_values))
    

@np.vectorize
def q(radial_coordinate: float,
      axis_q: float,
      shaping_exponent: float) -> float:
    r = radial_coordinate
    nu = shaping_exponent

    # Prevent division by zero for small r values.
    # For this function, in the limit r->0, q(r)->1. This is proven
    # mathematically in the lab book.
    #print(r)
    if np.abs(r) < 1e-5:
        return axis_q

    return axis_q*(nu+1)*(r**2)/(1-(1-r**2)**(nu+1))

def generate_q_profile(axis_q: float,
                       shaping_exponent: float) -> List[Tuple[float, float]]:
    """
    Generate a q-profile from the q() function defined above

    Returns
    -------
    List[Tuple[float, float]]
        List containing elements of the form (minor_radial_coord, q_at_coord).

    """
    r_values = np.linspace(0.0, 1.0, 100)
    q_values = q(r_values, axis_q, shaping_exponent)
    
    return list(zip(r_values, q_values))

def rational_surface(q_profile: List[Tuple[float, float]],
                     target_q: float) -> float:
    """
    Compute the location of the rational surface of the q-profile defined in q().
    """
    r, q = zip(*q_profile)
    func_r_of_q = UnivariateSpline(q, r, s=0)
    
    return func_r_of_q(target_q)

def rational_surface_of_mode(q_profile: List[Tuple[float, float]],
                             poloidal_mode: int,
                             toroidal_mode: int) -> float:
    # TODO: This function used to take axis_q and return an adjusted rational
    # surface with an axis_q correction. Make sure changing this hasn't 
    # messed things up elsewhere.
    
    return rational_surface(
        q_profile,
        float(poloidal_mode)/float(toroidal_mode)
    )

def magnetic_shear_profile(q_profile: List[Tuple[float, float]]):
    rs, qs = zip(*q_profile)
    q_spline = UnivariateSpline(rs, qs, s=0)
    dq_dr_spline = q_spline.derivative()
    
    q_at_r = q_spline(rs)
    dq_dr_at_r = dq_dr_spline(rs)  
    
    return rs*dq_dr_at_r/q_at_r

def magnetic_shear(q_profile: List[Tuple[float, float]],
                   r: float) -> float:
    """
    Calculate the magnetic shear of the plasma at r.

    s(r) = r*q'(r)/q(r)
    """
    rs, qs = zip(*q_profile)
    q_spline = UnivariateSpline(rs, qs, s=0)
    dq_dr_spline = q_spline.derivative()
    
    q_at_r = q_spline(r)
    dq_dr_at_r = dq_dr_spline(r)
    
    
    return r*dq_dr_at_r/q_at_r

if __name__=='__main__':
    from matplotlib import pyplot as plt
    
    q_profile = generate_q_profile(1.0, 2.0)
    r_s = rational_surface_of_mode(q_profile, 2, 1)
    print(magnetic_shear(q_profile, r_s))
    s_profile = magnetic_shear_profile(q_profile)
    r_vals, q_vals = zip(*q_profile)
    plt.plot(r_vals, s_profile)
