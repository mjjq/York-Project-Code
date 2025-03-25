import numpy as np
from matplotlib import pyplot as plt
import sys

if __name__=='__main__':
	files = sys.argv[1:]
	
	fig, ax = plt.subplots(1)

	for filename in files:
		file_data = np.loadtxt(filename)

		ax.plot(file_data[:,0], np.abs(file_data[:,1]/1e6))

	ax.set_xlabel("$\Psi_N$")
	ax.set_ylabel("Bootstrap current density (MA/m$^2$)")

	plt.show()
