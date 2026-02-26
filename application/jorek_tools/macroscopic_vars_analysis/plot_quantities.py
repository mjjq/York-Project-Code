from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from argparse import ArgumentParser
from typing import List, Optional, Tuple

from jorek_tools.jorek_dat_to_array import (
	read_postproc_profiles, PostprocProfile,
	read_timestep_map, TimestepMap
)


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
		self.x_errors: np.ndarray = None
		self.y_errors: np.ndarray = None

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
			
			try:
				values = mac_quantity[:,column_index]
			except IndexError:
				print(f"Couldn't get data for index {column_index}")
				values = None

			try:
				column_name = col_names[column_index]
			except IndexError:
				print(f"Couldn't get column name for index {column_index}")
				column_name = "None"

			return column_name, values
		
		raise ValueError("Failed to acquire data in _get_column_by_index")
	
	def load_x_values_by_index(self, column_index: int):
		self.x_val_name, self.x_values = self._get_column_by_index(column_index)

	def load_y_values_by_index(self, column_index: int):
		self.y_val_name, self.y_values = self._get_column_by_index(column_index)

	def load_x_errors_by_index(self, column_index: int):
		try:
			_, self.x_errors = self._get_column_by_index(column_index)
		except ValueError:
			self.x_errors = None

	def load_y_errors_by_index(self, column_index: int):
		try:
			_, self.y_errors = self._get_column_by_index(column_index)
		except ValueError:
			self.y_errors = None


def plot_macroscopic_quantities(quantities: List[PostprocProfile],
								labels: Optional[List[str]],
								time_map: Optional[TimestepMap],
								time_min: Optional[float],
								time_max: Optional[float],
								x_axis_label: Optional[str],
								y_axis_label: Optional[str],
								x_scale: str,
								y_scale: str,
								figure_size: Tuple[float, float],
								xmin: Optional[float],
								xmax: Optional[float],
								marker_style: str,
								marker_size: float,
								aspect_ratio: str,
								output_filename: Optional[str]):
	if figure_size is not None:
		fig, ax = plt.subplots(1, figsize=figure_size)
	else:
		fig, ax = plt.subplots(1)

	ax.set_aspect(aspect_ratio)
	xlabel = x_axis_label
	ylabel = y_axis_label

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	ax.grid(which='both')
	ax.set_yscale(y_scale)
	ax.set_xscale(x_scale)

	cmap = plt.cm.jet

	if labels is not None:
		for i,pair in enumerate(zip(labels, quantities)):
			label, mac_quantity = pair
			color = cmap(i/len(quantities))
			ax.errorbar(
				mac_quantity.x_vals,
				mac_quantity.y_vals,
				fmt=marker_style,
				label=label,
				capsize=2.0,
				markersize=marker_size,
				color=color
			)
		if len(quantities) > 1:
			ax.legend()
	else:
		color_vals = np.arange(0,len(quantities))
		cbar_label = "Normalised timestep"
		if time_map:
			q_timesteps = [q.timestep for q in quantities]
			q_times = np.interp(
				q_timesteps,
				time_map.time_steps,
				time_map.times
			)
			color_vals = q_times
			cbar_label = "Time (s)"

		min_color = min(color_vals)
		if time_min:
			min_color = time_min
		max_color = max(color_vals)
		if time_max:
			max_color = time_max
		normalise = mcolors.Normalize(
			vmin=min_color, vmax=max_color
		)

		for i,q in enumerate(quantities):
			c_val = color_vals[i]
			ax.errorbar(
				q.x_vals,
				q.y_vals,
				fmt=marker_style,
				capsize=2.0,
				markersize=marker_size,
				color=cmap(normalise(c_val))
			)

		if(len(quantities) > 1):
			scalarmappaple = cm.ScalarMappable(norm=normalise, cmap=cmap)
			scalarmappaple.set_array(color_vals)
			cbar = fig.colorbar(scalarmappaple, ax=ax)
			cbar.ax.set_ylabel(cbar_label)

	if xmin:
		ax.set_xlim(left=xmin)
	if xmax:
		ax.set_xlim(right=xmax)


	if 'lin' in x_scale:
		ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2), useOffset=True)
	if 'lin' in y_scale:
		ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useOffset=True)

	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
	plt.tight_layout()

	def on_resize(event):
		fig.tight_layout()
		fig.canvas.draw()
	cid = fig.canvas.mpl_connect("resize_event", on_resize)

	if output_filename:
		plt.savefig(output_filename, dpi=300, bbox_inches='tight')

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
	parser.add_argument('-yi', '--ycolumn-index', type=int, nargs="+", default=[1])
	parser.add_argument('-xerr', '--x-error-index', type=int, default=None)
	parser.add_argument('-yerr', '--y-error-index', type=int, default=None)
	parser.add_argument(
		'-xl', '--x-label', help="Name of x-axis quantity")
	parser.add_argument('-yl', '--y-label', help="Name of y-axis quantity")
	parser.add_argument(
		'-l', '--labels', nargs='+', help="Legend labels for each input file"
	)
	parser.add_argument(
		'-tm', '--time-map', type=str, help="Path to timestep-> time file", default=None
	)
	parser.add_argument(
		'-t0', '--time-min', type=float, help="Min time filter", default=None
	)
	parser.add_argument(
		'-t1', '--time-max', type=float, help="Max time filter", default=None
	)
	parser.add_argument('-x0', '--xmin', type=float, help='Minimum X-value to plot')
	parser.add_argument('-x1', '--xmax', type=float, help="Maximum X-value to plot")
	parser.add_argument(
		'-xs', '--x-scale', choices=['linear','log'], help="X-axis scale",
		default='linear'
	)
	parser.add_argument(
		'-ys', '--y-scale', choices=['linear','log'], help="Y-axis scale",
		default='linear'
	)
	parser.add_argument(
		'-fs', '--figure-size', nargs=2, type=float, help="Figure size (tuple)",
		default=None
	)
	parser.add_argument(
		'-t', '--marker-type', 
		help="Plotting marker type (x, -, etc)", default='-'
	)
	parser.add_argument(
		'-ms', '--marker-size', type=float,
		help="Plotting marker size (x, -, etc)", default=1.0
	)
	parser.add_argument('-a', '--aspect-ratio', help="Axes aspect ratio", default='auto')
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
			for yi in args.ycolumn_index:
				mq = read_postproc_profiles(filename, args.xcolumn_index, yi)
				quantities.append(mq)

	quantities = np.array(quantities).flatten()

	labels = None
	if args.labels:
		labels = args.labels

	time_map = None
	if args.time_map:
		time_map = read_timestep_map(args.time_map)

	plot_macroscopic_quantities(
		quantities,
		labels,
		time_map,
		args.time_min,
		args.time_max,
		args.x_label,
		args.y_label,
		args.x_scale,
		args.y_scale,
		args.figure_size,
		args.xmin,
		args.xmax,
		args.marker_type,
		args.marker_size,
		args.aspect_ratio,
		args.output_filename
	)

	#mq = MacroscopicQuantity()
