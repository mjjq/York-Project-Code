from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser
from typing import List, Optional


class MacroscopicQuantity:
	"""
	Parses a macroscopic variable .dat file generated using
	JOREK's util/extract_live_data.sh script, converts into
	a convenient class structure.
	"""
	def __init__(self, _filename: str):
		self.column_name: str
		self.filename: str = _filename
		self.times: np.ndarray
		self.column_vals: np.ndarray
		
		self.times: np.ndarray
		self.column_vals: np.ndarray


	def load_column_by_name(self, _column_name: str):
		"""
		Parse a macroscopic variable.dat file generated using JOREK's
		util/extract_live_data.sh script. Automatically extracts the
		column requested using column_name.
		"""
		self.column_name = _column_name

		mac_quantity_vs_time = np.loadtxt(self.filename, skiprows=1)
		self.times = mac_quantity_vs_time[:,0]
		
		with open(self.filename) as f:
			# Read first line which contains column info, remove
			# new line delimiter with rstrip()
			# Remove quote marks from column_names with replace
			first_line = f.readline().rstrip().replace('"','')
			col_names = np.array(list(filter(None, first_line.split(" "))))
			# Should only retrieve a single index since col names
			# should be unique. However, argwhere returns an array,
			# so flatten then retrieve first index.
			col_index = np.argwhere(col_names==_column_name).flatten()[0]
			self.column_vals = mac_quantity_vs_time[:,col_index]

	def load_column_by_index(self, column_index: int):
		# Ignore first row since this contains headers
		mac_quantity_vs_time = np.loadtxt(self.filename, skiprows=1)
		self.times = mac_quantity_vs_time[:,0]
		
		with open(self.filename) as f:
			# Read first line which contains column info, remove
			# new line delimiter with rstrip()
			# Remove quote marks from column_names with replace
			first_line = f.readline().rstrip().replace('"','')
			col_names = np.array(list(filter(None, first_line.split(" "))))
			
			self.column_vals = mac_quantity_vs_time[:,column_index]
			self.column_name = col_names[column_index]


def plot_macroscopic_quantities(quantities: List[MacroscopicQuantity],
								y_axis_label: str,
								y_scale: str,
								xmin: Optional[float],
								xmax: Optional[float],
								output_filename: Optional[str]):
	fig, ax = plt.subplots(1)
	ax.set_xlabel("Time (ms)")
	ax.set_ylabel(y_axis_label)

	ax.grid(which='both')
	ax.set_yscale(y_scale)


	for mac_quantity in quantities:
		ax.plot(
			mac_quantity.times,
			mac_quantity.column_vals,
			label=mac_quantity.column_name
		)

	if xmin:
		ax.set_xlim(left=xmin)
	if xmax:
		ax.set_xlim(right=xmax)

	ax.legend()

	if output_filename:
		plt.savefig(output_filename, dpi=300)

	plt.show()


if __name__ == "__main__":
	parser = ArgumentParser(
		prog="Macroscopic quantities plotter",
		description="Plots macroscopic quantity of a given column" \
			" against time for multiple JOREK runs",
		epilog="Note: Number of files must match number of columns! "
			"Alternatively, use the -ci option to choose a column to "
			"plot across all files."
	)
	parser.add_argument('-f', '--files',  nargs='+')
	parser.add_argument('-c', '--columns', nargs='+')
	parser.add_argument('-ci', '--column-index', type=int)
	parser.add_argument('-yl', '--y-label', help="Name of y-axis quantity")
	parser.add_argument('-x0', '--xmin', type=float, help='Minimum X-value to plot')
	parser.add_argument('-x1', '--xmax', type=float, help="Maximum X-value to plot")
	parser.add_argument(
		'-ys', '--y-scale', choices=['linear','log'], help="Y-axis scale",
		default='linear'
	)
	parser.add_argument('-o', '--output-filename', help="Output plot filename", default=None)
	args = parser.parse_args()

	quantities = []
	if args.columns:
		assert len(args.files)==len(args.columns), \
			"Number of columns must equal number of files!"
		for filename, column_name in zip(args.files, args.columns):
			mq = MacroscopicQuantity(filename)
			mq.load_column_by_name(column_name)
			quantities.append(mq)
	elif args.column_index:
		for filename in args.files:
			mq = MacroscopicQuantity(filename)
			mq.load_column_by_index(args.column_index)
			quantities.append(mq)


	plot_macroscopic_quantities(
		quantities,
		args.y_label,
		args.y_scale,
		args.xmin,
		args.xmax,
		args.output_filename
	)

	#mq = MacroscopicQuantity()
