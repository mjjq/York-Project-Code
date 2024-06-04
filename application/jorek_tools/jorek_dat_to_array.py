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

	Note: radial co-ordinate is normalised to minor radius, assume minor radius
	equals greatest radius in data.
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

    #print(data)
    
    psi_n, qs = zip(*data)
    plt.plot(psi_n, qs)

    # Take absolute q values since JOREK outputs negative q for some reason
    data = list(zip(psi_n, np.abs(qs)))

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
    
    # Normalise minor radial co-ordinate to minor radius of plasma
    rs, qs = zip(*q_r)
    q_r = list(zip(np.array(rs)/np.max(rs), qs))

    rs, js = zip(*j_r)
    j_r = list(zip(np.array(rs)/np.max(rs), js))

    return q_r, j_r

def q_and_j_from_csv(filename_exprs: str, filename_q:str) -> \
    Tuple[ List[Tuple[float, float]], List[Tuple[float, float]] ]:
        
    exprs_data: pd.DataFrame = pd.read_csv(filename_exprs)
    q_data = read_q_profile(filename_q)
    
    psi_n_data = list(zip(exprs_data['Psi_N'],exprs_data['r_minor']))
    q_r = map_q_to_rminor(psi_n_data, q_data)
    
    # Use zj as this seems to be non-normalised compared to currdens and Jphi
    j_r = list(zip(exprs_data['r_minor'], exprs_data['zj']))

    
    # Normalise minor radial co-ordinate to minor radius of plasma
    rs, qs = zip(*q_r)
    q_r = list(zip(np.array(rs)/np.max(rs), qs))

    rs, js = zip(*j_r)
    j_r = list(zip(np.array(rs)/np.max(rs), js))

    return q_r, j_r

def main():
    import sys

    if len(sys.argv) < 3:
        filename_exprs = "./postproc/exprs_averaged_s00000.csv"
        filename_q = "./postproc/qprofile_s00000.dat"
    else:
        filename_exprs = sys.argv[1]
        filename_q = sys.argv[2]
    q_r, j_r = q_and_j_from_csv(filename_exprs, filename_q)
    
    print(q_r, j_r)

def read_q_profile_temporal(q_filename: str) -> pd.DataFrame:
    #data = np.genfromtxt(q_filename, comments=None)
    #print(data)

    data = []

    with open(q_filename, 'r') as file:
        lines = []
        for line in file:
            if '#' in line:
                data.append(lines)
                lines = []
            lines.append(line)

    #print(data)
    headers = [d[0] if d else None for d in data]
    numbers = [np.abs(np.genfromtxt(d[1:])) if d else None for d in data]
    #print(headers)
    #print(numbers[1])

    #fig, ax = plt.subplots(1)
    profiles = pd.DataFrame()
    for i, d in enumerate(zip(headers, numbers)):
        header, qprof = d
        #print(qprof)
        if qprof is None:
            continue
        if qprof.size==0:
            continue
        psi_ns, qs = zip(*qprof)
        timestep = int(''.join(filter(str.isdigit, header)))

        df = pd.DataFrame({
            'timestep':[timestep]*len(psi_ns),
            'Psi_N':psi_ns,
            'q':qs
        })
        
        profiles = pd.concat([profiles, df])
        #ax.plot(rs, qs, label=header, color=(i/len(data), 0.0, 0.0))

    #ax.hlines(2.0, xmin=0.0, xmax=1.0, label='q=2', linestyle='--')
    #ax.legend()
    #ax.set_ylim(bottom=1.99, top=2.01)
    #plt.show()

    return profiles

def plot_profiles(profiles: pd.DataFrame):
    #print(profiles)
    pivoted = profiles.pivot('timestep', 'Psi_N')

    print(pivoted.columns)

def test():
    import sys
    q_filename = sys.argv[1]

    profiles = read_q_profile_temporal(q_filename)
    plot_profiles(profiles)

if __name__=='__main__':
    main()
