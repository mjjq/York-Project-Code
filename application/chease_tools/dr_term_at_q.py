from argparse import ArgumentParser
from numpy import interp, loadtxt

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


if __name__=='__main__':
	parser = ArgumentParser(
		description='Get D_R term at a given safety factor from CHEASE column data'
	)

	parser.add_argument('filename', type=str, nargs='+', help='Name of chease_cols file')
	parser.add_argument(
		'-q', '--safety-factor', type=float,
		help='Safety factor at which to evaluate D_R'
	)

	args = parser.parse_args()

	for fname in args.filename:
		print(extract_dr_from_cols(fname, args.safety_factor))
