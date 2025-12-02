from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser

import readxtor as rx


def extract_growth_rate(energy_filename: str,
			toroidal_mode_number: int):
	"""
	Extract growth rate from XTOR energy file for a given
	toroidal mode number.

	Growth rate is returned in terms of Alfven units. Note:
	Magnetic growth rate is returned, NOT the energy growth
	rate!

	:param energy_filename: Path to the XTOR simulation's energies.dat
	"""

	coords = plt.ginput(2)

	t0 = coords[0][0]
	t1 = coords[1][0]

	w0 = np.interp(t0, t, w_n)
	w1 = np.interp(t1, t, w_n)

	# Factor of 0.5 necessary to return magnetic growth
	# rate instead of energy growth rate.
	return 0.5*np.log(w1/w0)/(t1-t0)


if __name__=='__main__':
	parser = ArgumentParser()

	parser.add_argument(
		'energy_filenames', nargs='+',
		help='Paths to XTOR energy files'
	)
	parser.add_argument(
		'-n', '--toroidal-mode-number', type=int,
		help='Toroidal mode number to analyse',
		default=1
	)
	parser.add_argument(
		'-l', '--labels', nargs='+',
		help='Labels associated with each energy file'
	)

	args = parser.parse_args()

	energy_filenames = args.energy_filenames
	toroidal_mode_number = args.toroidal_mode_number

	fig, ax = plt.subplots(1)
	ax.grid()
	ax.set_xlabel(r'Time ($\tau_a$)')
	ax.set_ylabel(r'Magnetic energy (arb)')
	ax.set_yscale('log')
	for i, filename in enumerate(energy_filenames):
		t,k,w,p = rx.ReadEnergies(filename)

		w_n = w[:,toroidal_mode_number]

		label=""
		if args.labels:
			label = args.labels[i]

		ax.plot(t, w_n, label=label)

	ax.legend()
	plt.show()
