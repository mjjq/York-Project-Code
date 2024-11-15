from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser
from typing import List


class MacroscopicQuantity:
	"""
	Parses a macroscopic variable .dat file generated using
	JOREK's util/extract_live_data.sh script, converts into
	a convenient class structure.
	"""
	def __init__(self, filename: str, column_name: str):
		self.column_name: str = column_name
		self.filename: str = filename
		self.times: np.ndarray
		self.column_vals: np.ndarray
		
		self.times, self.column_vals = self.load_quantity_from_file(
			filename, column_name
		)

	def load_quantity_from_file(self, _filename: str, _column_name: str):
		"""
		Parse a macroscopic variable.dat file generated using JOREK's
		util/extract_live_data.sh script. Automatically extracts the
		column requested using column_name.
		"""
		mac_quantity_vs_time = np.loadtxt(_filename, skiprows=1)
		times = mac_quantity_vs_time[:,0]
		
		with open(_filename) as f:
			# Read first line which contains column info, remove
			# new line delimiter with rstrip()
			# Remove quote marks from column_names with replace
			first_line = f.readline().rstrip().replace('"','')
			col_names = np.array(list(filter(None, first_line.split(" "))))
			# Should only retrieve a single index since col names
			# should be unique. However, argwhere returns an array,
			# so flatten then retrieve first index.
			col_index = np.argwhere(col_names==_column_name).flatten()[0]
			col_vals = mac_quantity_vs_time[:,col_index]

		return times, col_vals


def plot_macroscopic_quantities(quantities: List[MacroscopicQuantity],
								y_axis_label: str):
	fig, ax = plt.subplots(1)
	ax.set_xlabel("Time [JOREK units]")
	ax.set_ylabel(y_axis_label)

	for mac_quantity in quantities:
		ax.plot(
			mac_quantity.times,
			mac_quantity.column_vals,
			label=mac_quantity.column_name
		)

	ax.legend()

	plt.show()


if __name__ == "__main__":
	parser = ArgumentParser(
		prog="Macroscopic quantities plotter",
		description="Plots macroscopic quantity of a given column" \
			" against time for multiple JOREK runs",
		epilog="Note: Number of files must match number of columns!"
	)
	parser.add_argument('-f', '--files',  nargs='+')
	parser.add_argument('-c', '--columns', nargs='+')
	parser.add_argument('-q', '--quantity')
	args = parser.parse_args()

	print(args.files, args.columns)

	assert len(args.files)==len(args.columns), "Number of columns/files mismatch, exiting"

	quantities = []
	for filename, column_name in zip(args.files, args.columns):
		quantities.append(MacroscopicQuantity(filename, column_name))

	plot_macroscopic_quantities(quantities, args.quantity)

	#mq = MacroscopicQuantity()