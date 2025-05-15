import numpy as np
import sys
import pandas as pd
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from typing import List, Tuple
import os
import re
from io import StringIO
from dataclasses import dataclass

from jorek_tools.dat_to_pandas import dat_to_pandas

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

def read_eta_profile_r_minor(postproc_exprs_filename: str) -> List[Tuple[float, float]]:
    """
    Get resistivity from postproc output as a function of minor radius
    """
    exprs_data = dat_to_pandas(postproc_exprs_filename)

    return list(zip(exprs_data['r_minor'], exprs_data['eta_T']))


def read_R0(postproc_exprs_filename: str) -> float:
    """
    Get Major radius (geometric, r_minor=0) from postproc output
    """
    exprs_data = dat_to_pandas(postproc_exprs_filename)

    return list(exprs_data['R'])[0]

def read_Btor(postproc_exprs_filename: str) -> float:
    """
    Get on-axis toroidal field from postproc output
    """
    exprs_data = dat_to_pandas(postproc_exprs_filename)

    return list(exprs_data['Btor'])[0]

def read_r_minor(postproc_exprs_filename: str) -> float:
    """
    Get minor radius of the plasma from postproc output
    """
    exprs_data = dat_to_pandas(postproc_exprs_filename)

    return list(exprs_data['r_minor'])[-1]


def read_rho0(postproc_exprs_filename: str) -> float:
    """
    Get central mass density of the plasma from postproc output
    """
    exprs_data = dat_to_pandas(postproc_exprs_filename)

    return list(exprs_data['rho'])[0]

def read_chi_perp_profile_rminor(postproc_exprs_filename: str) \
    -> List[Tuple[float, float]]:
    """
    Get perpendicular heat diffusivity profile in terms of minor radius
    """
    exprs_data = dat_to_pandas(postproc_exprs_filename)

    return list(zip(exprs_data['r_minor'], exprs_data['zkprof']))


def read_chi_par_profile_rminor(postproc_exprs_filename: str) \
    -> List[Tuple[float, float]]:
    """
    Get parallel heat diffusivity profile in terms of minor radius
    """
    exprs_data = dat_to_pandas(postproc_exprs_filename)

    return list(zip(exprs_data['r_minor'], exprs_data['zkpar_T']))



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
        
    exprs_data: pd.DataFrame = dat_to_pandas(filename_exprs)
    q_data = read_q_profile(filename_q)
    
    psi_n_data = list(zip(exprs_data['Psi_N'],exprs_data['r_minor']))
    q_r = map_q_to_rminor(psi_n_data, q_data)
    
    # Use JOREK's normalised zj (non-SI units). Tested this numerically
    # in lab book in 2024, section "On the appropriateness of zj"
    j_r = list(zip(exprs_data['r_minor'], exprs_data['zj']))

    
    # Normalise minor radial co-ordinate to minor radius of plasma
    rs, qs = zip(*q_r)
    q_r = list(zip(np.array(rs)/np.max(rs), qs))

    rs, js = zip(*j_r)

    j_r = list(zip(np.array(rs)/np.max(rs), np.abs(js)))

    return q_r, j_r

@dataclass
class Four2DProfile:
    poloidal_mode_number: int
    toroidal_mode_number: int
    psi_norm: np.ndarray
    r_minor: np.ndarray
    psi: np.ndarray

def read_four2d_profile(four2d_filename: str) -> List[Four2DProfile]:
    """
    Read a four2D output file and return arrays of 
    Psi_N vs Psi for all mode numbers present.
    """
    split_data = []
    with open(four2d_filename, 'r') as f:
        raw_data = f.read()

        # Data for each m/n mode is split by a 
        # double line-break separated by a space
        # i.e. "\n \n"
        split_data = raw_data.split("\n \n")

    ret: List[Four2DProfile] = []
    for raw_profile in split_data:
        # Some regex magic.
        poloidal_mode_str, toroidal_mode_str = re.findall(
            r'[+-]\d{3}', raw_profile
        )

        profile_data = np.loadtxt(StringIO(raw_profile))
        psi_n_data = profile_data[:,0]
        r_minor_data = profile_data[:,1]
        psi_data = profile_data[:,2]

        profile: Four2DProfile = Four2DProfile(
            poloidal_mode_number=int(poloidal_mode_str),
            toroidal_mode_number=int(toroidal_mode_str),
            r_minor=r_minor_data,
            psi_norm=psi_n_data,
            psi=psi_data
        )

        ret.append(profile)

    return ret

def read_four2d_profile_filter(four2d_filename: str,
                               poloidal_mode_number: int):
    try:
        return list(filter(
            lambda x: x.poloidal_mode_number==poloidal_mode_number,
            read_four2d_profile(four2d_filename)
        ))[0]
    except IndexError:
        raise ValueError(
            f"Could not find data for m={poloidal_mode_number}"
        )


def main():
    import sys

    four2d_filename = sys.argv[1]

    from matplotlib import pyplot as plt
    prof = list(filter(
        lambda x: x.poloidal_mode_number==2,
        read_four2d_profile(four2d_filename)
    ))[0]

    plt.plot(prof.psi_norm, prof.psi)
    plt.show()

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
