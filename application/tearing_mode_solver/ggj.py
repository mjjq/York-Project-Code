import sympy as sp
import numpy as np

def solve_complex_eq(A: float,
                     B: float) -> complex:
    t = sp.symbols('t')

    eq = A*t**6 - t - A*B

    solutions_t = sp.nroots(eq)

    solutions_z = np.array([complex(sol)**4 for sol in solutions_t])

    return solutions_z

def critical_d_r(lundquist_number: float,
			    delta_prime_ext: float,
			    magnetic_shear: float,
			    poloidal_mode_number: float,
			    toroidal_mode_number: float,
			    r_s: float) -> float:
    """
	Calculate the critical curvature stabilisation D_R term 
    (inverse of critical_delta_prime)
	(fluid version where diffusion is unimportant (GGJ 1975 result).
	Full criterion taken from Brunetti's MHD report, eq. 17.56

	Convention the same as in :func curvature_stabilisation_kinetic:
	
	:param lundquist_number: The lundquist number at the rational surface
	:param delta_prime_ext: External Delta'
	:param magnetic_shear: Magnetic shear at the rational surface
	:param poloidal_mode_number: Poloidal mode number
	:param toroidal_mode_number: Toroidal mode number
	:param r_s: Radius of rational surface (normalised to minor radius)
	
	:return Curvature stabilisation Delta' contribution normalised to minor radius
    """
    # Result of 2*pi*Gamma(3/4)/Gamma(1/4) * (pi/4)**(5/6) 
    #            * cos(pi/4)/cos(pi/8) * (cot(pi/8)**(1/6)
    const_terms = 1.53929937748165595101
    q_s = poloidal_mode_number/toroidal_mode_number

    return (
        r_s * delta_prime_ext *
        (1+2.0*q_s**2)**(1/6) /
        const_terms /
        (toroidal_mode_number*magnetic_shear*lundquist_number)**(1/3)
    )**(6/5)

def solve_ggj_dispersion_relation(delta_prime_ext: float,
                                  poloidal_mode_number: int,
                                  toroidal_mode_number: int,
                                  r_s: float,
                                  shear_rs: float,
                                  lundquist_number: float,
                                  resistive_interchange: float) -> float:
    """
    Solve the full GGJ dispersion relation (GGJ 1975, eq. 88).

    We use r_s V_s/X_0 = (nsS)^(1/3)/(1+2q_s^2)^(1/6) (Brunetti MHD report)

    :param delta_prime_ext: External Delta' calculated in the ideal
    outer region
    :param poloidal_mode_number: Poloidal mode number
    :param toroidal_mode_number: Toroidal mode number
    :param r_s: Radius of rational surface
    :param shear_rs: Magnetic shear at the rational surface
    :param lundquist_number: The lundquist number
    :param q_s: Safety factor at the rational surface

    :return growth rate normalised to Alfven frequency
    """
    crit_dr = critical_d_r(
        lundquist_number,
        delta_prime_ext,
        shear_rs,
        poloidal_mode_number,
        toroidal_mode_number,
        r_s
    )

    if np.abs(resistive_interchange) >= crit_dr:
        return 0.0

    # Gamma(3/4)/Gamma(1/4)
    gam = 0.33798912003364236449772384233540287414364172745770297598843145700274869783

    A=(
        2.0*np.pi/(r_s*delta_prime_ext) * 
        (toroidal_mode_number*shear_rs*lundquist_number)**(1/3) /
        (1.0+2.0*(poloidal_mode_number/toroidal_mode_number)**2)**(1/6) *
        gam
    )

    B = 0.25*np.pi*resistive_interchange

    Q = solve_complex_eq(A,B)

    gr = Q*lundquist_number**(-1/3)


    # Find solutions where imaginary component is zero
    # Multiple solutions may exist so choose largest
    # magnitude.
    # If none exist, search for complex solution with
    # smallest imaginary component
    ims = np.array([np.abs(num.imag) for num in gr])
    zero_ims = np.where(ims**2 < 1e-15)
    if np.any(zero_ims):
        real_sols = gr[zero_ims]
        return np.max(real_sols)
    else:
        min_ims = np.argmin(ims)
        return gr[min_ims].real


def ntm_ggj_term(w: float,
                 d_r: float,
                 beta_p: float,
                 w_d: float = 0) -> float:
    """
    Calculate GGJ contribution to the MRE.

    See Kleiner NF 2016 equation 10.

    We normalise island width to minor radius. Delta' is
    normalised to minor radius also, i.e. we return a*Delta'.

    Parameters
    ------------
    :param w: Magnetic island width normalised to minor radius
    :param d_r: Resistive interchange parameter at the rational
        surface
    :param beta_p: Poloidal beta at the rational surface
    :param w_d: Correction due to finite diffusion width 
    """
    return 6.0 * d_r / beta_p * (
        w/(w**2 + 0.2*(w_d)**2)
    )



if __name__=='__main__':
    gams = []
    d_r_vals = -np.logspace(-8, -2, 20)#[-0.00182, -0.002, -0.003, -0.00368, -0.0037]
    for d_r in d_r_vals:
        gams.append( solve_ggj_dispersion_relation(
            1.5, 2, 1, 0.5, 1.0, 5.50e6, d_r
        ))
        
    from matplotlib import pyplot as plt
    plt.plot(np.abs(d_r_vals), gams, 'x-')
    plt.show()
