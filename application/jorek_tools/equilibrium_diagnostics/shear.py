from scipy.interpolate import splrep, splev
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from tearing_mode_solver.profiles import rational_surface

if __name__=='__main__':
	parser = ArgumentParser()
	parser.add_argument('qprofile_filename', type=str, help="Path to q-profile")
	parser.add_argument('-p','--plot',action='store_true', help='Plot shear profile')
	parser.add_argument('-q', '--target-q', type=float, default=2.0, help='Target q to evaluate shear')
	parser.add_argument('-r', '--r-range', nargs=2, help='Radial range to filter q-profile', default=(0.02, 0.98))

	args = parser.parse_args()

	q_data = np.loadtxt(args.qprofile_filename)

	psi_n, q, r = zip(*q_data)

	r = np.array(r)
	q = np.abs(np.array(q))

	r_min, r_max = args.r_range

	r_filter = (r>r_min) & (r<r_max)

	r = r[r_filter]
	q = q[r_filter]

	r_s = rational_surface(list(zip(r,q)), args.target_q)

	q_spline = splrep(r, q,k=4,s=1)
	q_prime_vals = splev(r, q_spline,der=1)
	spline_shear_vals = r*q_prime_vals/q

	q_prime = np.diff(q)/np.diff(r)

	r = r[:-1]
	q = q[:-1]

	shear_vals = r*q_prime/q

	print("\n".join([f"{x} {y}" for x,y in zip(r, shear_vals)]))

	if(args.plot):
		fig, ax = plt.subplots(1, figsize=(4,4))
		ax.plot(r, shear_vals,label='shear', alpha=0.4)
		ax.plot(r, spline_shear_vals[:-1],label='shear (smoothed)')
		ax.set_xlabel('r (m)')
		ax.set_ylabel('s(r)')
		ax.vlines(r_s, min(shear_vals), max(shear_vals), label=r'$r_s$(q='f'{args.target_q})', color='black')
		ax.legend()
		ax.grid()
		plt.tight_layout()
		plt.show()
