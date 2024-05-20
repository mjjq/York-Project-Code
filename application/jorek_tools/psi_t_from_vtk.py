import vtk
from vtk import vtkUnstructuredGridReader
from vtk.util import numpy_support as vn
from vtk.util.numpy_support import vtk_to_numpy

import glob
import sys
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator

def print_point_data_attributes(vtk_point_data):
	for idx in range(vtk_point_data.GetNumberOfArrays()):
		array = vtk_point_data.GetArray(idx)
		print(array.GetName(), array.GetDataType(), vtk_to_numpy(array).dtype)

def extract_psi_t(filenames: List[str]) -> pd.DataFrame:
	"""
	Extract maximum perturbed flux from JOREK VTK files for each timestep and
	return the data as a tuple of (array(times), array(psi)).
	"""
	reader = vtkUnstructuredGridReader()

	#filenames = glob.glob(filename_glob)
	#print(filenames)
	out = pd.DataFrame()
	for filename in filenames:
		reader.SetFileName(filename)

		reader.ReadAllScalarsOn()
		reader.ReadAllVectorsOn()
		reader.Update()

		pdata = reader.GetOutput().GetPointData()
		print_point_data_attributes(pdata)
		
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
		
		interp = CloughTocher2DInterpolator(cyl_coords, psi_values)

		theta_sample_vals = np.linspace(0.0, 2.0*np.pi, 100)
		r_sample_vals = np.linspace(0.0, max(radii), 100)
		max_psi_r = [max(interp(r, theta_sample_vals)) for r in r_sample_vals]

		time = reader.GetOutput().GetFieldData().GetArray("TIME")
		time = max(vtk_to_numpy(time))

		times = [time]*len(r_sample_vals)

		df = pd.DataFrame({
			'time':times,
			'r':r_sample_vals,
			'Psi':max_psi_r
		})
		out = pd.concat([out, df])
		#print(time, max_psi)

	return out

#def export_psi_t_to_csv(times: np.array, psi_t: np.array, filename: str):


if __name__=='__main__':
	fnames=sys.argv[1:]

	print(fnames)

	psi_data = extract_psi_t(fnames)

	psi_data.to_csv("psi_t_data.csv")

	group = psi_data.groupby("time")
	group.plot(x='r', y='Psi')
	from matplotlib import pyplot as plt

	#fig, ax = plt.subplots(1)
	#ax.plot(times, psi_t, color='black')
	#ax.set_xlabel(r"Time $(1/\omega_A)$")
	#ax.set_ylabel(r"Flux $(a^2 B_{\phi 0})$")

	plt.show()
