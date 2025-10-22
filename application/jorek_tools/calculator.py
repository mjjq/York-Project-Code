import numpy as np
import sys

# Physical Constants (CODATA 2018)

c = 299792458             # speed of light in vacuum, m/s (exact)
mu_0 = 4e-7*np.pi   # vacuum permeability (N/A^2)
epsilon_0 = 8.8541878188e-12  # vacuum permittivity (F/m)
e = 1.602176634e-19       # elementary charge (C)
k = 1.380649e-23          # Boltzmann constant (J/K)
N_A = 6.02214076e23       # Avogadro constant (1/mol)
R = 8.314462618           # molar gas constant (J/(mol⋅K))
m_e = 9.1093837015e-31    # electron mass (kg)
m_p = 1.67262192369e-27   # proton mass (kg)
m_n = 1.67492749804e-27   # neutron mass (kg)

if __name__=='__main__':
	args = sys.argv[1]

	print(eval(args))
