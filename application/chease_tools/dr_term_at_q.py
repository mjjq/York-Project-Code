from argparse import ArgumentParser
import numpy as np
from numpy import interp, loadtxt, pi
from dataclasses import dataclass

@dataclass
class CheaseColumns():
	"""
	Partial conversion of CHEASE raw numpy data
	to columns.

	TODO: Implement remaining columns
	"""
	s: np.array # s co-ordinate
	q: np.array # Safety factor
	p: np.array # Pressure
	dp_dpsi: np.array # dP/dPsi
	shear: np.array # Magnetic shear
	b_avg: np.array # <B>
	eps: np.array # inverse aspect ratio
	d_r: np.array # resistive interchange
	d_i: np.array # Ideal mercier interchange
	shift_prime: np.array # Shafranov shift radial derivative
	F: np.array # F=RB_phi (F=T in CHEASE)

def read_columns(filename: str) -> CheaseColumns:
	"""
	Read raw chease columns from file and store into
	convenient class
	"""
	raw_data = loadtxt(filename, comments="%")

	return CheaseColumns(
		s=raw_data[:,0],
		q=raw_data[:,7],
		p=raw_data[:,5],
		dp_dpsi=raw_data[:,6],
		shear=raw_data[:,9],
		b_avg=raw_data[:,29],
		eps=raw_data[:,44],
		d_r=raw_data[:,-7],
		d_i=raw_data[:,80],
		shift_prime=raw_data[:,60],
		F=raw_data[:,3]
	)


def extract_dr_from_cols(cols: CheaseColumns, q: float) -> float:
	"""
	Extract resistive interchange (D_R) parameter from chease
	column data.

	:param cols: CHEASE column data
	:param q: Safety factor

	:return: Resistive interchange value at q
	"""
	q_column = cols.q
	dr_column = cols.d_r

	# RESISTIVE INTERCHANGE column gives -D_R
	# Hence, must negate to get D_R
	d_r_at_q = -interp(q, q_column, dr_column)

	return d_r_at_q


def alpha_from_cols(cols: CheaseColumns, q: float) -> float:
	"""
	Extract normalised ballooning parameter (normalised
	pressure gradient) at a given q-surface.

	We take expression 12 in Graves 2019:

	alpha = -2q^2 R0/B0^2 dP/dr.

	Using dpsi/dr = rF/(qR0)

	and dP/dpsi = P_c' * B0/R0^2
	F = F_c * R0 * B0,

	one arrives at the normalised form of alpha for
	CHEASE:

	alpha = -2q*eps(r) * F_c * P_c'

	:param cols: CHEASE column data
	:param q: Safety factor

	:return: Normalised pressure gradient at q
	"""
	eps = cols.eps
	q_column = cols.q
	dp_dpsi_column = cols.dp_dpsi

	dp_dpsi = interp(q, q_column, dp_dpsi_column)
	eps = interp(q, q_column, eps)
	F=interp(q, q_column, cols.F)

	alpha = -(2.0*q) * eps * F * dp_dpsi
	return alpha

def d_i_approximation_from_cols(cols: CheaseColumns, q: float) -> float:
	"""
	Calculate large aspect ratio approximation to D_R from CHEASE
	columns, assuming no plasma shaping.

	:param cols: CHEASE column data
	:param q: Safety factor at which to evaluate D_R

	:return: Large aspect D_R approximation
	"""
	alpha = alpha_from_cols(cols, q)
	shear = interp(q, cols.q, cols.shear)
	eps = cols.eps[-1]


	return alpha * eps / shear**2 * (1/q**2 - 1.0) - 0.25

def extract_rmhd_dr_from_cols(cols: CheaseColumns, q: float) -> float:
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

	:param cols: CHEASE column data
	:param q: Safety factor at which to evaluate D_R

	:return: Reduced MHD modification to D_R value at q
	"""
	shear = interp(q, cols.q, cols.shear)

	alpha = alpha_from_cols(cols, q)
	eps = interp(q, cols.q, cols.eps)

	delta_d_r = - alpha**2 / (4.0*shear**2 * q**2) - eps*alpha/(shear**2 * q**2)

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
	#parser.add_argument(
	#	'-a', '--approximate', action='store_true',
	#	help='Return large aspect ratio D_R approximation instead of CHEASE calculated'
	#)

	args = parser.parse_args()

	for fname in args.filename:
		cols = read_columns(fname)

		d_r = extract_dr_from_cols(cols, args.safety_factor)
		#if args.approximate:
		#	d_r = d_i_approximation_from_cols(cols, args.safety_factor)

		if args.reduced_mhd:
			delta_dr = extract_rmhd_dr_from_cols(cols, args.safety_factor)
			d_r = d_r + delta_dr

		print(f"{d_r:.10f}")
