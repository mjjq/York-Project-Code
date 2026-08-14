from typing import List, Tuple
import numpy as np

from jorek_tools.poincare.read_poincare_vtk import read_poincare_vtk
from jorek_tools.jorek_dat_to_array import PostprocProfile,read_postproc_profiles

Vec2D = Tuple[float, float]

def cross_product_2d(vec_a: Vec2D, 
                     vec_b: Vec2D) -> float:
    ax, ay = vec_a
    bx, by = vec_b

    return ax*by - ay*bx

def dot_product_2d(vec_a: Vec2D, 
                   vec_b: Vec2D) -> float:
    ax, ay = vec_a
    bx, by = vec_b

    return ax*bx + ay*by

def angle_between_two_vectors(vec_a: Vec2D,
                              vec_b: Vec2D) -> float:
    """
    Get generalised angle between two vectors in the domain 0 <= theta < 2pi

    Vectors must be normalised to length 1!
    """
    c_prod = cross_product_2d(vec_a, vec_b)
    d_prod = dot_product_2d(vec_a, vec_b)

    # Consider right handed co-ordinates, theta going anti-clockwise
    # If angle in top left quadrant
    if (c_prod >= 0.0) and (d_prod >= 0.0):
        return np.acos(d_prod)
    # If angle in bottom left quadrant
    elif (c_prod >= 0.0) and (d_prod < 0.0):
        return np.acos(d_prod)
    # If angle in bottom right quadrant
    elif (c_prod < 0.0) and (d_prod < 0.0):
        return 2.0*np.pi - np.acos(d_prod)
    # If angle in top right quadrant
    else:
        return 2.0*np.pi - np.acos(d_prod)


def q_profile(poincare_in: List[PostprocProfile]) -> List[PostprocProfile]:
    # We assume the first surface is directly on the magnetic axis, or
    # at least very close to it. Average all points on this surface to 
    # get the location
    first_surface = poincare_in[0]
    r_axis = np.mean(first_surface.x_vals)
    z_axis = np.mean(first_surface.y_vals)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)

    #poincare_in = [poincare_in[1], poincare_in[8]]
    q = []
    for i,surface in enumerate(poincare_in):
        r, z = surface.x_vals, surface.y_vals

        if len(r)==0:
            continue
        delta_r = r-r_axis
        delta_z = z-z_axis

        theta = np.atan2(delta_z, delta_r)
        theta = np.unwrap(theta)
        
        n_tor_turns = len(theta)-1
        n_pol_turns = (theta[-1]-theta[0])/(2*np.pi)

        #print((n_tor_turns, n_pol_turns))

        q.append(n_tor_turns/n_pol_turns)

    plt.show()

    return np.array(q)

if __name__=='__main__':
    import sys

    fname = sys.argv[1]

    if ".vtk" in fname:
        poincare = read_poincare_vtk(fname)
    else:
        poincare = read_postproc_profiles(fname)

    print(q_profile(poincare))
