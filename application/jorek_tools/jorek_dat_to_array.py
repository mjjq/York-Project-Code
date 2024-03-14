import numpy as np
import sys
import pandas as pd
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from typing import List, Tuple

def read_psi_profile(filename: str) -> List[Tuple[float, float]]:
    """
    Get raw minor radius values as a function of Psi_N from postproc output.

    Returns a list where each element is a tuple of (psi_n_at_r, r_minor_coord).
    """
    data = np.genfromtxt(
        filename
    )
    psi_n, r_minor, currdens = zip(*data)
    
    #fig, ax = plt.subplots(2)
    #ax[0].plot(r_minor, psi_n)
    #ax[1].plot(r_minor, currdens)

    return list(zip(psi_n, r_minor))

def read_j_profile(filename: str) -> List[Tuple[float, float]]:
	"""
	Get raw current density profile values as a function of r_minor from postproc
	output.

	Returns a list where each element is a tuple of (r_minor_coord, currdens)
	"""
	data = np.genfromtxt(filename)

	psi_n, r_minor, currdens = zip(*data)

    # Get absolute current density, JOREK returns negative current for some
    # reason.
	return list(zip(r_minor, np.abs(currdens)))

def read_q_profile(filename: str) -> List[Tuple[float, float]]:
    """
    Get raw safety factor profile values as a function of Psi_N from postproc
    output. 

    We later use this to derive q as a function of r_minor
    """
    data = np.genfromtxt(filename)
    
    psi_n, qs = zip(*data)
    plt.plot(psi_n, qs)

    return data

def map_q_to_rminor(psi_n_data: List[Tuple[float, float]],
		    q_data: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
	"""
	Map q(psi) to r(psi) to return q(r). Assumes the flux profile is cylindrically
	symmetric. Will not work for a shaped plasma.
	"""
	psi_n, r_minor = zip(*psi_n_data)

	r_as_func_of_psi = UnivariateSpline(psi_n, r_minor, s=0.0)
	# Note: Take absolute value of q because JOREK outputs negative values
	
	q_psi_n, q_vals = zip(*q_data)
	q_as_func_of_psi = UnivariateSpline(q_psi_n, np.abs(q_vals), s=0.0)

	out_array = [
		(float(r_as_func_of_psi(p)), float(q_as_func_of_psi(p))) 
		for p in psi_n
	]

	return out_array

def q_and_j_from_input_files(filename_psi: str, filename_q: str) -> \
    Tuple[ List[Tuple[float, float]], List[Tuple[float, float]] ]:
    psi_data = read_psi_profile(filename_psi)
    q_data = read_q_profile(filename_q)
    
    q_r = map_q_to_rminor(psi_data, q_data)
    
    j_r = read_j_profile(filename_psi)
    
    return q_r, j_r


if __name__=='__main__':
    filename_psi = sys.argv[1]
    psi_data = read_psi_profile(filename_psi)
    j_data = read_j_profile(filename_psi)
    #print(psi_data)

    filename_q = sys.argv[2]
    q_data = read_q_profile(filename_q)
    #print(q_data)

    q_r = map_q_to_rminor(psi_data, q_data)
    #print(q_r)
    
    from matplotlib import pyplot as plt
    #rs, qs = zip(*q_r)
    #print(rs)
    rs, qs = zip(*q_r)
    rjs, js = zip(*j_data)
    
    fig, ax = plt.subplots(2)
    ax[0].plot(rs, qs)
    ax[1].plot(rjs, js)

    fig2, ax2 = plt.subplots(1)
    
    ax2.plot(rjs, (1-np.array(rjs)**2)**2)