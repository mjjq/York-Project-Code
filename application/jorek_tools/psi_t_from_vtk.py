import vtk
from vtk import vtkUnstructuredGridReader
from vtk.util import numpy_support as vn
from vtk.util.numpy_support import vtk_to_numpy

from tqdm import tqdm
import sys
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator, UnivariateSpline

def print_point_data_attributes(vtk_point_data):
	for idx in range(vtk_point_data.GetNumberOfArrays()):
		array = vtk_point_data.GetArray(idx)
		print(array.GetName(), array.GetDataType(), vtk_to_numpy(array).dtype)

def extract_psi_t_single(filename: str, 
						 reader: Optional[vtkUnstructuredGridReader] = None) \
	-> pd.DataFrame:
	"""
	Extract maximum peturbed flux from single JOREK VTK file
	"""
	if reader is None:
		reader = vtkUnstructuredGridReader()

	reader.SetFileName(filename)

	reader.ReadAllScalarsOn()
	reader.ReadAllVectorsOn()
	reader.Update()

	pdata = reader.GetOutput().GetPointData()
#		print_point_data_attributes(pdata)
	
	coords = reader.GetOutput().GetPoints().GetData()
	coords = vtk_to_numpy(coords)

	# Assume first point is by definition the central point of the mesh
	centre = coords[0]
	coords_com = np.array([c-centre for c in coords])
	radii = np.sqrt(np.array([(c[0]**2 + c[1]**2) for c in coords_com]))
	
	xs, ys, zs = zip(*coords_com)
	thetas = np.arctan2(ys, xs)

	cyl_coords = list(zip(radii, thetas))

	psi_values = pdata.GetArray("Psi")
	psi_values = vtk_to_numpy(psi_values)
	
	max_delta_psi_arg = np.argmax(psi_values)
	max_r, max_theta = cyl_coords[max_delta_psi_arg]

	interp_delta_psi = CloughTocher2DInterpolator(cyl_coords, psi_values)

	psi_norms = pdata.GetArray("psi_norm")
	psi_norms = vtk_to_numpy(psi_norms)
	interp_psi_norm = CloughTocher2DInterpolator(cyl_coords, psi_norms)

	#theta_sample_vals = np.linspace(0.0, 2.0*np.pi, 100)
	r_sample_vals = np.linspace(0.0, max(radii), 100)
	#max_psi_r = [
	#	max(interp_delta_psi(r, theta_sample_vals)) for r in r_sample_vals
	#]
	max_psi_r = [interp_delta_psi(r, max_theta) for r in r_sample_vals]
	max_psi_norms = [interp_psi_norm(r, max_theta) for r in r_sample_vals]

	time = reader.GetOutput().GetFieldData().GetArray("TIME")
	time = max(vtk_to_numpy(time))

	times = [time]*len(r_sample_vals)

	df = pd.DataFrame({
		'filename':filename,
		'time':times,
		'r':r_sample_vals,
		'Psi':max_psi_r,
		'psi_norm':max_psi_norms
	})

	return df

def extract_psi_t(filenames: List[str],
				  output_to_file: bool = False,
				  output_filename: Optional[str] = None) -> pd.DataFrame:
	"""
	Extract maximum perturbed flux from JOREK VTK files for each timestep and
	return the data as a tuple of (array(times), array(psi)).

	Append to file after each iteration as each iter is slow and if the program
	terminates early it can resume from where it left off this way.
	"""
	reader = vtkUnstructuredGridReader()

	out = pd.DataFrame(
		columns=['filename','time','r','Psi','psi_norm']
	)
	if output_to_file:
		try:
			out = pd.read_csv(output_filename)
			print("Existing file found, will append to this one")
		except FileNotFoundError:
			print("Existing file not found, creating new.")
			out.to_csv(output_filename)

	#filenames = glob.glob(filename_glob)
	#print(filenames)
	print(out['filename'])
	remaining_files = list(set(filenames)-set(out['filename']))
	pbar = tqdm(remaining_files)
	for filename in pbar:
		#print(filename)
		pbar.set_description(filename)

		df = extract_psi_t_single(filename, reader)
		if output_to_file:
			df.to_csv(output_filename, mode='a', header=False)

		out = pd.concat([out, df])
		#print(time, max_psi)

	return out


def jorek_flux_interp_func(jorek_psi_t_data: pd.DataFrame) \
    -> CloughTocher2DInterpolator:
    """
    Get temporal evolution of flux at a particular radial co-ordinate.

    Parameters
    ----------
    jorek_psi_t_data : pd.DataFrame
        Dataframe containing perturbed flux as a function of r and t.
        
    Returns
    -------
    CloughTocher2DInterpolator
        Psi(r, t)

    """
    grouped = jorek_psi_t_data.groupby('time')
    
    vals = []
    coords = []
    
    for time, group in grouped:
        vals += list(group['Psi'])
        coords += [(time, r_val) for r_val in group['r']]
        
    vals = np.array(vals)
    coords = np.array(coords)
    # TODO: Output is noisy, try to fix this!
    return CloughTocher2DInterpolator(coords, vals, maxiter=400, rescale=True)
        
def r_from_q(q_profile: List[Tuple[float, float]],
             target_q: float):
    rs, qs = zip(*q_profile)
    
    spline = UnivariateSpline(qs, rs, s=0)
    
    return spline(target_q)
    

def jorek_flux_at_q(jorek_data: pd.DataFrame,
                    q_profile: List[Tuple[float, float]],
                    target_q: float) -> Tuple[List[float], List[float]]:
    target_r = r_from_q(q_profile, target_q)
    
    jorek_psi_t_func = jorek_flux_interp_func(jorek_data)
    
    times = np.unique(jorek_data['time'])
    
    return np.array(times), \
        np.array([jorek_psi_t_func(t, target_r) for t in times])
 


if __name__=='__main__':
	fnames=sys.argv[1:]

	print(fnames) 

	psi_data = extract_psi_t(fnames, True, "psi_t_data.csv")

	# print("Saving to psi_t_data.csv")
	# try:
	# 	psi_data.to_csv("psi_t_data.csv")
	# 	print("Saved successfully!")
	# except Exception as e:
	# 	print(f"Failed to save. Reason {e}")

	#group = psi_data.groupby("time")
	#group.plot(x='r', y='psi_norm')
	#from matplotlib import pyplot as plt

	#fig, ax = plt.subplots(1)
	#ax.plot(times, psi_t, color='black')
	#ax.set_xlabel(r"Time $(1/\omega_A)$")
	#ax.set_ylabel(r"Flux $(a^2 B_{\phi 0})$")

	#plt.show()
