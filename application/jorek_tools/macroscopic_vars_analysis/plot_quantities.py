from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser
from typing import List, Optional, Tuple


class MacroscopicQuantity:
	"""
	Parses a macroscopic variable .dat file generated using
	JOREK's util/extract_live_data.sh script, converts into
	a convenient class structure.
	"""
	def __init__(self, _filename: str):
		self.filename: str = _filename
		self.x_values: np.ndarray
		self.y_values: np.ndarray

		self.x_val_name: str
		self.y_val_name: str


	def _get_column_by_name(self, _column_name: str) -> np.ndarray:
		"""
		Parse a macroscopic variable.dat file generated using JOREK's
		util/extract_live_data.sh script. Automatically extracts the
		column requested using column_name.
		"""
		mac_data = np.loadtxt(self.filename, skiprows=1)
		#self.x_values = mac_quantity_vs_time[:,0]
		
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

			return mac_data[:,col_index]
		
		raise ValueError("Failed to acquire data in _get_column_by_name")

	def load_x_values_by_name(self, _column_name: str):
		self.x_values = self._get_column_by_name(_column_name)
		self.x_val_name = _column_name

	def load_y_values_by_name(self, _column_name: str):
		self.y_values = self._get_column_by_name(_column_name)
		self.y_val_name = self._get_column_by_name

	def _get_column_by_index(self, column_index: int) -> Tuple[str, np.ndarray]:
		# Ignore first row since this contains headers
		mac_quantity = np.loadtxt(self.filename, skiprows=1)
		
		with open(self.filename) as f:
			# Read first line which contains column info, remove
			# new line delimiter with rstrip()
			# Remove quote marks from column_names with replace
			first_line = f.readline().rstrip().replace('"','')
			col_names = np.array(list(filter(None, first_line.split(" "))))
			
			values = mac_quantity[:,column_index]
			column_name = col_names[column_index]

			return column_name, values
		
		raise ValueError("Failed to acquire data in _get_column_by_index")
	
	def load_x_values_by_index(self, column_index: int):
		self.x_val_name, self.x_values = self._get_column_by_index(column_index)

	def load_y_values_by_index(self, column_index: int):
		self.y_val_name, self.y_values = self._get_column_by_index(column_index)


def plot_macroscopic_quantities(quantities: List[MacroscopicQuantity],
								labels: Optional[List[str]],
								x_axis_label: Optional[str],
								y_axis_label: Optional[str],
								y_scale: str,
								xmin: Optional[float],
								xmax: Optional[float],
								output_filename: Optional[str]):
	fig, ax = plt.subplots(1)
	
	xlabel = x_axis_label
	if xlabel is None:
		xlabel = quantities[0].x_val_name
	ylabel = y_axis_label
	if ylabel is None:
		ylabel = quantities[0].y_val_name

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	ax.grid(which='both')
	ax.set_yscale(y_scale)

	if labels is None:
		labels = [mq.y_val_name for mq in quantities]
	for label, mac_quantity in zip(labels, quantities):
		ax.plot(
			mac_quantity.x_values,
			mac_quantity.y_values,
			label=label
		)

	if xmin:
		ax.set_xlim(left=xmin)
	if xmax:
		ax.set_xlim(right=xmax)

	if len(quantities) > 1:
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
	parser.add_argument('-xi', '--xcolumn-index', type=int, default=0)
	#parser.add_argument('-c', '--columns', nargs='+')
	parser.add_argument('-yi', '--ycolumn-index', type=int, default=1)
	parser.add_argument(
		'-xl', '--x-label', help="Name of x-axis quantity")
	parser.add_argument('-yl', '--y-label', help="Name of y-axis quantity")
	parser.add_argument(
		'-l', '--labels', nargs='+', help="Legend labels for each input file"
	)
	parser.add_argument('-x0', '--xmin', type=float, help='Minimum X-value to plot')
	parser.add_argument('-x1', '--xmax', type=float, help="Maximum X-value to plot")
	parser.add_argument(
		'-ys', '--y-scale', choices=['linear','log'], help="Y-axis scale",
		default='linear'
	)
	parser.add_argument('-o', '--output-filename', help="Output plot filename", default=None)
	args = parser.parse_args()

	quantities = []
	# if args.columns:
	# 	assert len(args.files)==len(args.columns), \
	# 		"Number of columns must equal number of files!"
	# 	for filename, column_name in zip(args.files, args.columns):
	# 		mq = MacroscopicQuantity(filename)
	# 		mq.load_x_values_by_index(args.xcolumn)
	# 		mq.load_y_values_by_name(column_name)
	# 		quantities.append(mq)
	if args.ycolumn_index:
		for filename in args.files:
			mq = MacroscopicQuantity(filename)
			mq.load_x_values_by_index(args.xcolumn_index)
			mq.load_y_values_by_index(args.ycolumn_index)
			quantities.append(mq)

	labels = None
	if args.labels:
		labels = args.labels

	plot_macroscopic_quantities(
		quantities,
		labels,
		args.x_label,
		args.y_label,
		args.y_scale,
		args.xmin,
		args.xmax,
		args.output_filename
	)

	#mq = MacroscopicQuantity()
