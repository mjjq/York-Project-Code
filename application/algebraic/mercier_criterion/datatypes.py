from dataclasses import dataclass
import sympy as sp

@dataclass
class ToroidalCoordinates:
    R: sp.Function # Major radius
    Z: sp.Function # Vertical distance

    D: sp.Function # Shafranov shift (function of r)
    P: sp.Function # P(r)

    R0: sp.Symbol # Major radius of magnetic axis
    
    # New co-ords
    r: sp.Symbol # Minor radius
    omega: sp.Symbol # Generalised angle

    # Expansion params
    epsilon: sp.Symbol # Inverse aspect ratio tag
    ellip: sp.Symbol # Ellipticity tag

@dataclass
class EPFLShaping:
    # Shaping
    S: sp.Function # Shaping coefficients (S_m(r))
    m: sp.Symbol # Shaping coef indices
    N: sp.Symbol # sum_{m=1}^N...

@dataclass
class LutjensShaping:
    E: sp.Function # ellipticity E(r)
    T: sp.Function # Triangularity T(r)


@dataclass
class Derivatives:
    R_r: sp.Function
    R_omega: sp.Function
    Z_r: sp.Function
    Z_omega: sp.Function


@dataclass
class MetricTensor:
    g_rr: sp.Function
    g_rw: sp.Function
    g_ww: sp.Function


@dataclass
class GradShafVariables:
    gs: sp.Function # Equation A.3. in Lutjens 1992 expanded

    f: sp.Function # dpsi/dr = R0 B0 eps * f(r)
    g: sp.Function # G = R0 B0 g(r), g(r) = 1+O(eps^2)

    p: sp.Function # pressure p(r)

    B0: sp.Symbol # On-axis toroidal field