import vtk
from vtk import vtkUnstructuredGridReader
from vtk.util import numpy_support as vn
from vtk.util.numpy_support import vtk_to_numpy

import glob
import sys
from typing import List, Tuple
import numpy as np
import pandas as pd

def extract_psi_t(filenames: List[str]) -> Tuple[np.array, np.array]:
	"""
	Extract maximum perturbed flux from JOREK VTK files for each timestep and
	return the data as a tuple of (array(times), array(psi)).
	"""
	reader = vtkUnstructuredGridReader()

	#filenames = glob.glob(filename_glob)
	#print(filenames)
	times = []
	psi_t = []
	for filename in filenames:
		reader.SetFileName(filename)

		reader.Update()

		psi_values = reader.GetOutput().GetPointData().GetArray("Psi")
		psi_values = vtk_to_numpy(psi_values)
		max_psi = max(psi_values)

		time = reader.GetOutput().GetFieldData().GetArray("TIME")
		time = max(vtk_to_numpy(time))

		times.append(time)
		psi_t.append(max_psi)
		print(time, max_psi)

	return times, psi_t

#def export_psi_t_to_csv(times: np.array, psi_t: np.array, filename: str):


if __name__=='__main__':
	fnames=sys.argv[1:]

	print(fnames)

	times, psi_t = extract_psi_t(fnames)

	df = pd.DataFrame({
		'times':times,
		'psi_t': psi_t
	})
	df.to_csv("psi_t_data.csv")

	from matplotlib import pyplot as plt

	fig, ax = plt.subplots(1)
	ax.plot(times, psi_t, color='black')
	ax.set_xlabel(r"Time $(1/\omega_A)$")
	ax.set_ylabel(r"Flux $(a^2 B_{\phi 0})$")

	plt.show()
