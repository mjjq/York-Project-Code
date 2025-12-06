from argparse import ArgumentParser
from numpy import interp, loadtxt, pi

def extract_dr_from_cols(filename: str, q: float) -> float:
	"""
	Extract resistive interchange (D_R) parameter from chease
	column data.

	:param filename: Name of the chease_cols file
	:param q: Safety factor

	:return: Resistive interchange value at q
	"""
	cols_data = loadtxt(filename, comments='%')

	# q-profile column located at index 7
	q_column = cols_data[:,7]
	# D_R column located at index -7
	dr_column = cols_data[:,-7]

	# RESISTIVE INTERCHANGE column gives -D_R
	# Hence, must negate to get D_R
	d_r_at_q = -interp(q, q_column, dr_column)

	return d_r_at_q

def extract_rmhd_dr_from_cols(filename: str, q: float) -> float:
	"""
	Calculated reduced MHD modification to D_R from chease column
	data.

	See Graves 2019 modification to D_R in reduced MHD

	Note: CHEASE already normalised dP/dpsi (see Lutjens
	paper), and so mu0/B0^2 factor has been removed
	from alpha calcualtion.

	CHEASE also gives calculation in terms of dP/dpsi
	but we need dP/dr. Note that s=r/a = sqrt(psi),
	so dP/dpsi = dP/ds * ds/dpsi = a dP/dr * 0.5/s.

	:param filename: Name of the chease_cols file
	:param q: Safety factor at which to evaluate D_R

	:return: Reduced MHD modification to D_R value at q
	"""
	cols_data = loadtxt(filename, comments='%')

	B0 = cols_data[0,29]
	eps = cols_data[-1,44]

	q_column = cols_data[:,7]
	s_coord_column = cols_data[:,0]
	dp_dpsi_column = cols_data[:,6]
	shear_column = cols_data[:,9]

	s = interp(q, q_column, s_coord_column)
	dp_dpsi = interp(q, q_column, dp_dpsi_column)
	shear = interp(q, q_column, shear_column)

	alpha = -(4.0*q**2 / eps) * s * dp_dpsi

	delta_d_r = - alpha**2 / (4.0*shear**2 * q**2)

	return delta_d_r


if __name__=='__main__':
	parser = ArgumentParser(
		description='Get D_R term at a given safety factor from CHEASE column data'
	)

	parser.add_argument('filename', type=str, nargs='+', help='Name of chease_cols file')
	parser.add_argument(
		'-q', '--safety-factor', type=float,
		help='Safety factor at which to evaluate D_R'
	)
	parser.add_argument(
		'-r', '--reduced-mhd', action='store_true',
		help='Return reduced-MHD D_R if enabled'
	)

	args = parser.parse_args()

	for fname in args.filename:
		d_r = extract_dr_from_cols(fname, args.safety_factor)

		if args.reduced_mhd:
			delta_dr = extract_rmhd_dr_from_cols(fname, args.safety_factor)
			d_r = d_r + delta_dr

		print(f"{d_r:.10f}")
