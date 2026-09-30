from argparse import ArgumentParser
import numpy as np
from numpy import interp, loadtxt, pi
from dataclasses import dataclass
from typing import List, Tuple
from itertools import cycle
from matplotlib import pyplot as plt

from rdcon_tools.delta_gw import time_from_g_filename, shot_num_from_g_filename

def read_columns_raw(filename: str) -> np.array:
    """
    Read a chease_cols.out file and return the raw
    data
    """
    return loadtxt(filename, comments="%")

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
    beta_p: np.array # Poloidal beta (propto p/<Bp>**2)
    r_avg: np.array # Poloidally averaged major radius
    r_inboard: np.array
    r_outboard: np.array
    # Poloidally averaged bootstrap current 
    # (choose the zerocoll version)
    j_bs: np.array
    # Poloidally averaged j_phi (not including j_bs)
    j_phi: np.array
    # Lower triangularity
    delta_bottom: np.array
		
def read_columns(filename: str) -> CheaseColumns:
	"""
	Read raw chease columns from file and store into
	convenient class
	"""
	raw_data = read_columns_raw(filename)

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
		F=raw_data[:,3],
		beta_p=raw_data[:,75],
		r_avg=raw_data[:,25],
		j_bs=raw_data[:,33],
		j_phi=raw_data[:,10],
        r_inboard=raw_data[:,63],
        r_outboard=raw_data[:,64],
        delta_bottom=raw_data[:,66]
	)

def time_averaged_raw_cols(files: List[str]) -> Tuple[np.array, np.array]:
    """
    Generate a time-average of the raw chease columns.
    The first return value is the raw array of mean values.
    The second is the raw array of standard error values.
    """
    raw_col_array = np.array([
        read_columns_raw(f) for f in files
    ])

    mean_vals = np.mean(raw_col_array, axis=0)
    std_vals = np.std(raw_col_array, axis=0)/np.sqrt(raw_col_array.shape[0])

    return mean_vals, std_vals

def avg_cols_to_file(files: List[str], out_prefix: str = ""):
    mean_array, std_array = time_averaged_raw_cols(files)

    with open(files[0]) as f:
        header = f.readline().strip('\n')

        np.savetxt(f"{out_prefix}chease_cols_mean.out", mean_array, header=header, comments="%")
        np.savetxt(f"{out_prefix}chease_cols_std.out",  std_array,  header=header, comments="%")

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


def extract_term_at_q(raw_cols: np.array, 
                      term_index: int, 
                      q: float) -> float:
    """
    Read a given chease column term at a specified
    q-surface

    :param raw_cols: Raw CHEASE columns
    :param term_index: Index of the quantity to evaluate
    :param q: Safety factor at which to evaluate the
        quantity
    """
    # q_profile is index 7
    q_profile = raw_cols[:,7]

    quantity = raw_cols[:,term_index]

    quantity_at_q = interp(q, q_profile, quantity)

    return quantity_at_q

def alpha_from_cols(cols: CheaseColumns) -> float:
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
	q = cols.q
	dp_dpsi_column = cols.dp_dpsi

	dp_dpsi = dp_dpsi_column
	eps = eps
	F=cols.F

	alpha = -(2.0*q) * eps * F * dp_dpsi
	return alpha

def alpha_at_q(cols: CheaseColumns, q: float) -> float:
	return interp(q, cols.q, alpha_from_cols(cols))

def d_i_approximation_from_cols(cols: CheaseColumns, q: float) -> float:
	"""
	Calculate large aspect ratio approximation to D_R from CHEASE
	columns, assuming no plasma shaping.

	:param cols: CHEASE column data
	:param q: Safety factor at which to evaluate D_R

	:return: Large aspect D_R approximation
	"""
	alpha = alpha_at_q(cols, q)
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

	alpha = alpha_at_q(cols, q)
	eps = interp(q, cols.q, cols.eps)

	delta_d_r = - alpha**2 / (4.0*shear**2 * q**2) - eps*alpha/(shear**2 * q**2)

	return delta_d_r

@dataclass
class AvgCheaseProfs:
    psi_n: np.array
    d_r_avg: np.array 
    d_r_std: np.array 
    d_i_avg: np.array 
    d_i_std: np.array 
    shift_prime_avg: np.array 
    shift_prime_std: np.array 
    alpha_avg: np.array 
    alpha_std: np.array 
    shear_avg: np.array 
    shear_std: np.array
    q_avg: np.array
    q_std: np.array
    j_b_norm_avg: np.array
    j_b_norm_std: np.array
    delta_prime_ratio_avg: np.array
    delta_prime_ratio_std: np.array
    shot: int

def dr_avg_profile(files: List[str], tmin: float, tmax: float):
    cols_array = np.array([read_columns(fname) for fname in files])
    time_array = np.array([time_from_g_filename(fname) for fname in files])

    time_filt = (time_array > tmin) & (time_array < tmax)

    d_r_profs = np.array([col.d_r for col in cols_array])
    d_r_profs = d_r_profs[time_filt]

    d_i_profs = np.array([col.d_i for col in cols_array])
    d_i_profs = d_i_profs[time_filt]

    alpha_profs = np.array([alpha_from_cols(col) for col in cols_array])
    alpha_profs = alpha_profs[time_filt]

    shear_profs = np.array([col.shear for col in cols_array])
    shear_profs = shear_profs[time_filt]

    d_r_norm = d_r_profs

    d_r_avg = np.mean(d_r_norm, axis=0)
    d_r_std = np.std(d_r_norm, axis=0)/np.sqrt(len(d_r_norm))

    alpha_avg = np.mean(alpha_profs, axis=0)
    alpha_std = np.std(alpha_profs, axis=0)/np.sqrt(len(alpha_profs))

    shear_avg = np.mean(shear_profs, axis=0)
    shear_std = np.std(shear_profs, axis=0)/np.sqrt(len(shear_profs))

    # Note: Hinrich's thesis has a slightly different definition
    # for D_I than CosteSarguet2025. There is an offset of +1/4
    # in Hinrich's definition, so we subtract that here.
    # Additionally, the profiles from CHEASE are in terms of -D_I
    # which is why we have d_i_profs-0.25 and not -d_i_profs - 0.25
    d_i_norm = shear_profs**2 * (d_i_profs-0.25) / alpha_profs
    d_i_avg = np.mean(d_i_norm, axis=0)
    d_i_std = np.std(d_i_norm, axis=0)/np.sqrt(len(d_i_norm))

    q_profs = np.array([col.q for col in cols_array])
    q_profs = q_profs[time_filt]
    q_avg = np.mean(q_profs, axis=0)
    q_std = np.std(q_profs, axis=0)/np.sqrt(len(q_profs))

    eps_profs = np.array([col.eps for col in cols_array])
    eps_profs = eps_profs[time_filt]


    shift_prime_prof = np.array([col.shift_prime for col in cols_array])[time_filt]
    tria_prof = np.array([col.delta_bottom for col in cols_array])[time_filt]
    tria_prime_prof = (
        np.diff(tria_prof,append=0,axis=1)/
        np.diff(eps_profs,append=1,axis=1)
    )
    # The D'(r) measured by chease uses the experimental definition, see
    # https://gitlab.epfl.ch/spc/chease/-/blob/master/src-f90/output.f90#L390
    # Their definition is D_exp(r) = R_geo(r) - R_geo(a)
    # We convert to the analytic definition as per Jon's lecture notes
    # (Lecture 2, slide 42):
    # D_exp(r) = D(a)-D(r) + 0.25*(r*delta(r)-r*delta(a)), where 
    # delta(r) is the triangularity profile.
    shift_prime_analytic = -shift_prime_prof + 0.25*(eps_profs*tria_prof + tria_prime_prof)
    
    shift_prime_avg = np.mean(shift_prime_analytic, axis=0)
    shift_prime_std = np.std(shift_prime_analytic, axis=0)/np.sqrt(len(shift_prime_analytic))

    f_profs = np.array([col.F for col in cols_array])
    f_profs = f_profs[time_filt]

    j_b_profs = np.array([col.j_bs for col in cols_array])

    j_b_profs = j_b_profs[time_filt]
    j_b_norm = 0.5*(j_b_profs / shear_profs) / eps_profs / f_profs**2

    j_b_norm_avg = np.mean(j_b_norm, axis=0)
    j_b_norm_std = np.std(j_b_norm, axis=0)/np.sqrt(len(j_b_norm))

    delta_prime_ratio_profs = d_r_profs / j_b_norm
    delta_prime_ratio_avg = np.mean(delta_prime_ratio_profs, axis=0)
    delta_prime_ratio_std = np.std(delta_prime_ratio_profs, axis=0)/np.sqrt(len(delta_prime_ratio_profs))

    s_prof = cols_array[0].s
    psi_n_prof = s_prof**2

    shot = shot_num_from_g_filename(files[0])
    #from matplotlib import pyplot as plt
    #fig, ax = plt.subplots(1)
    #ax.plot(psi_n_prof, d_r_avg)
    #ax.fill_between(psi_n_prof, d_r_avg-d_r_std, d_r_avg+d_r_std, alpha=0.5)
    #plt.show()

    return AvgCheaseProfs(
        psi_n_prof, 
        d_r_avg, d_r_std, 
        d_i_avg, d_i_std,
        shift_prime_avg, shift_prime_std,
        alpha_avg, alpha_std, 
        shear_avg, shear_std, 
        q_avg, q_std, 
        j_b_norm_avg, j_b_norm_std,
        delta_prime_ratio_avg, delta_prime_ratio_std,
        shot
    )

def plot_avg_with_std(ax, psi_n: np.array, prof_avg: np.array, prof_std: np.array, label: str):
    # prof_avg[np.isnan(prof_avg)] = 0.0
    # prof_std[np.isnan(prof_std)] = 0.0
    try:
	    ax.plot(psi_n, prof_avg, label=label)
	    ax.fill_between(psi_n, prof_avg-prof_std, prof_avg+prof_std, alpha=0.2)
    except TypeError:
        raise ValueError(f"Bad profiles: {prof_avg}\n{prof_std}")

def plot_avg_profs(avg_profs: List[AvgCheaseProfs], 
                   q_s: float = 2.0, 
                   psi_min: float = 0.54, 
                   psi_max: float = 0.6):
    fig, ax = plt.subplots(5, sharex=True,gridspec_kw={"wspace": 0, "hspace": 0.2})
    ax_s, ax_alpha, ax_dr, ax_jb, ax_ratio = ax
    ax[-1].set_xlabel("$\psi_N$")

    ax_dr.set_ylabel(r"$-\hat{\Delta}_{GGJ}$")
    ax_jb.set_ylabel(r"$\hat{\Delta}_{BS}$")
    ax_ratio.set_ylabel(r"$-\hat{\Delta}_{GGJ}/\hat{\Delta}_{BS}$")
    ax_alpha.set_ylabel(r"$\alpha$")
    ax_s.set_ylabel(r"$s$")

    for ax_in in ax:
        #ax_in.grid()
        ax_in.set_xlim(psi_min, psi_max)

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for avg_prof in avg_profs:
        q_s_avg = interp(q_s, avg_prof.q_avg, avg_prof.psi_n)
        q_s_max = interp(q_s, avg_prof.q_avg+avg_prof.q_std, avg_prof.psi_n)
        q_s_min = interp(q_s, avg_prof.q_avg-avg_prof.q_std, avg_prof.psi_n)

        psi_filt = (
            (avg_prof.psi_n >= 0.98*psi_min) & 
            (avg_prof.psi_n <= 1.02*psi_max)
        )

        plot_avg_with_std(
            ax_dr,
            avg_prof.psi_n[psi_filt],
            avg_prof.d_r_avg[psi_filt],
            avg_prof.d_r_std[psi_filt],
            label=str(avg_prof.shot)
        )
        plot_avg_with_std(
            ax_jb,
            avg_prof.psi_n[psi_filt],
            avg_prof.j_b_norm_avg[psi_filt],
            avg_prof.j_b_norm_std[psi_filt],
            label=str(avg_prof.shot)
        )
        plot_avg_with_std(
            ax_ratio,
            avg_prof.psi_n[psi_filt],
            avg_prof.delta_prime_ratio_avg[psi_filt],
            avg_prof.delta_prime_ratio_std[psi_filt],
            label=str(avg_prof.shot)
        )
        # plot_avg_with_std(
        #     ax_ratio_norm,
        #     avg_prof.psi_n[psi_filt],
        #     avg_prof.q_avg[psi_filt],
        #     avg_prof.q_std[psi_filt],
        #     label=str(avg_prof.shot)
        # )
        plot_avg_with_std(
            ax_alpha,
            avg_prof.psi_n[psi_filt],
            avg_prof.alpha_avg[psi_filt],
            avg_prof.alpha_std[psi_filt],
            label=str(avg_prof.shot)
        )
        plot_avg_with_std(
            ax_s,
            avg_prof.psi_n[psi_filt],
            avg_prof.shear_avg[psi_filt],
            avg_prof.shear_std[psi_filt],
            label=str(avg_prof.shot)
        )

        color = next(colors)
        for ax_in in ax:
            ax_in.axvline(
                q_s_avg, 
                linestyle='--', 
                label=f"q=2 ({avg_prof.shot})",
                color=color
            )
    
    ax[0].legend(loc="upper left", bbox_to_anchor=(0,1.5), ncol=4)
    fig.tight_layout()


def plot_di_ss(avg_profs: List[AvgCheaseProfs], 
                q_s: float = 2.0, 
                psi_min: float = 0.55, 
                psi_max: float = 0.6):
    """
    Plot normalised ideal interchange (s^2 (-D_I-0.25) / alpha) and the
    derivative of the shafranov shift w.r.t psi_n
    """
    fig, ax = plt.subplots(2, sharex=True,gridspec_kw={"wspace": 0, "hspace": 0.2})
    ax_di, ax_ss = ax
    ax[-1].set_xlabel("$\psi_N$")

    ax_di.set_ylabel(r"$-s^2 D_I/ \alpha$")
    ax_ss.set_ylabel(r"$\Delta'_{ss}$")

    for ax_in in ax:
        #ax_in.grid()
        ax_in.set_xlim(psi_min, psi_max)

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for avg_prof in avg_profs:
        q_s_avg = interp(q_s, avg_prof.q_avg, avg_prof.psi_n)
        q_s_max = interp(q_s, avg_prof.q_avg+avg_prof.q_std, avg_prof.psi_n)
        q_s_min = interp(q_s, avg_prof.q_avg-avg_prof.q_std, avg_prof.psi_n)

        psi_filt = (
            (avg_prof.psi_n >= 0.98*psi_min) & 
            (avg_prof.psi_n <= 1.02*psi_max)
        )

        plot_avg_with_std(
            ax_di,
            avg_prof.psi_n[psi_filt],
            avg_prof.d_i_avg[psi_filt],
            avg_prof.d_i_std[psi_filt],
            label=str(avg_prof.shot)
        )
        plot_avg_with_std(
            ax_ss,
            avg_prof.psi_n[psi_filt],
            avg_prof.shift_prime_avg[psi_filt],
            avg_prof.shift_prime_std[psi_filt],
            label=str(avg_prof.shot)
        )

        color = next(colors)
        for ax_in in ax:
            ax_in.axvline(
                q_s_avg, 
                linestyle='--', 
                label=f"q=2 ({avg_prof.shot})",
                color=color
            )
    
    ax[0].legend(loc="upper left", bbox_to_anchor=(0,1.2), ncol=4)
    fig.tight_layout()


def avg_q2_radius(files: List[str], tmin: float, tmax: float):
    cols_array = np.array([read_columns(fname) for fname in files])
    time_array = np.array([time_from_g_filename(fname) for fname in files])

    q2_locs = np.array([np.interp(2.0, col.q, col.s) for col in cols_array])
    
    time_filt = (time_array > tmin) & (time_array < tmax)
    q2_locs_filt = q2_locs[time_filt]

    q2_psin = q2_locs_filt**2

    q2_avg = np.median(q2_psin)
    q2_std = np.std(q2_psin)/np.sqrt(len(q2_psin))

    print(q2_avg, q2_std)


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
    parser.add_argument(
        '-t', '--print-times', action='store_true',
        help="Print g-file times alongside output (requires eqdsk formatted folder names)"
    )
    parser.add_argument(
        '-a', '--average-profile', type=float, nargs=2, default=(None, None),
        help="Print average D_R profile over all g_files"
    )
    parser.add_argument(
        '-aq', '--average-q2-radius', type=float, nargs=2, default=(None, None),
        help="Print average q=2 radius over"
    )
    parser.add_argument(
            '-ao', '--output-average', action='store_true', 
            help="Save averaged profiles to disk"
    )
	#parser.add_argument(
	#	'-a', '--approximate', action='store_true',
	#	help='Return large aspect ratio D_R approximation instead of CHEASE calculated'
	#)

    args = parser.parse_args()

    if np.all(args.average_profile):
        tmin, tmax = args.average_profile
        shots = list(sorted(set([shot_num_from_g_filename(f) for f in args.filename])))
        avg_profs = []
        for shot in shots:
            files = [f for f in args.filename if str(shot) in f]
            avg_profs.append(dr_avg_profile(files, tmin, tmax))
            if args.output_average:
                avg_cols_to_file(files, out_prefix=f"{shot}_")
        plot_avg_profs(avg_profs)
        plot_di_ss(avg_profs)
        plt.show()
        exit()

    if np.all(args.average_q2_radius):
        tmin, tmax = args.average_q2_radius
        avg_q2_radius(args.filename, tmin, tmax)
        exit()

    for fname in args.filename:
        cols = read_columns(fname)

        d_r = extract_dr_from_cols(cols, args.safety_factor)
        #if args.approximate:
        #	d_r = d_i_approximation_from_cols(cols, args.safety_factor)

        if args.reduced_mhd:
            delta_dr = extract_rmhd_dr_from_cols(cols, args.safety_factor)
            d_r = d_r + delta_dr

        if args.print_times:
            time = time_from_g_filename(fname)
            print(f"{time} {d_r:.10f}")
        else:
            print(f"{d_r:.10f}")
