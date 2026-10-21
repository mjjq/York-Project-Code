import sympy as sp
from dataclasses import dataclass

def expand_small_parameter():
    epsilon = sp.symbols('epsilon')
    f = sp.exp(epsilon)/(1+epsilon)

    print(sp.series(f, epsilon, 0, 3))

def expand_small_param_x():
    a, x, epsilon = sp.symbols('a x epsilon')
    f = sp.sqrt(1 + a*epsilon*x)

    print(sp.series(f, epsilon, 0, 3))

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

def RZ_epfl() -> tuple[ToroidalCoordinates, EPFLShaping]:
    r, omega, epsilon, ellip, R0 = sp.symbols(
        "r omega epsilon ellip R0"
    )
    D = sp.Function("D")
    P = sp.Function("P")

    # For fourier summation
    m, N = sp.symbols("m N")
    # Fourier coefficients
    S = sp.Function("S")
    cos_expr = sp.Sum(S(m,r) * sp.cos((m-1)*omega), (m, 2, N))
    sin_expr = sp.Sum(S(m,r) * sp.sin((m-1)*omega), (m, 2, N))

    R = (
        R0 + epsilon * r * sp.cos(omega) - epsilon**2 * D(r) + 
        epsilon**2 * cos_expr + 
        epsilon**3 * P(r) * sp.cos(omega)
    )
    Z = (
        epsilon * r * sp.sin(omega) -
        epsilon**2 * sin_expr +
        epsilon**3 * P(r) * sp.sin(omega)
    )

    coords = ToroidalCoordinates(
        R, Z, D, P, R0, r, omega, epsilon, ellip
    )
    shaping = EPFLShaping(
        S, m, N
    )
    return coords, shaping

def RZ_lutjens() -> ToroidalCoordinates:
    r, omega, epsilon, ellip, R0 = sp.symbols(
        "r omega epsilon ellip R0"
    )
    D = sp.Function("D")
    E = sp.Function("E")
    T = sp.Function("T")
    P = sp.Function("P")

    R = (
        R0 - epsilon**2 * D(r) - epsilon*(r-E(r)) * sp.cos(omega) + 
        epsilon**2 * T(r) * sp.cos(2*omega) + 
        epsilon**3 * P(r) * sp.cos(3*omega)
    )
    Z = (
        epsilon * (r+ellip*E(r)) * sp.sin(omega) +
        epsilon**2 * T(r) * sp.sin(2*omega) -
        epsilon**3 * P(r) * sp.sin(omega)
    )

    return ToroidalCoordinates(
        R, Z, r, omega, epsilon, ellip, R0, P
    )

@dataclass
class Derivatives:
    R_r: sp.Function
    R_omega: sp.Function
    Z_r: sp.Function
    Z_omega: sp.Function

def calc_derivatives(coords: ToroidalCoordinates) -> Derivatives:
    return Derivatives(
        sp.diff(coords.R, coords.r)/coords.epsilon,
        sp.diff(coords.R, coords.omega)/coords.epsilon,
        sp.diff(coords.Z, coords.r)/coords.epsilon,
        sp.diff(coords.Z, coords.omega)/coords.epsilon
    )

@dataclass
class MetricTensor:
    g_rr: sp.Function
    g_rw: sp.Function
    g_ww: sp.Function

def calc_metric_tensor(derivatives: Derivatives) -> MetricTensor:
    g_rr = derivatives.R_r**2 + derivatives.Z_r**2
    g_rw = (
        derivatives.R_r*derivatives.R_omega + 
        derivatives.Z_r*derivatives.Z_omega
    )
    g_ww = derivatives.R_omega**2 + derivatives.Z_omega**2

    return MetricTensor(
        g_rr, g_rw, g_ww
    )

def calc_jacobian(coords: ToroidalCoordinates, 
                  derivatives: Derivatives) -> sp.Function:
    return coords.R * (
        derivatives.R_r * derivatives.Z_omega -
        derivatives.R_omega * derivatives.Z_r
    )

if __name__=='__main__':
    coords, shaping = RZ_epfl()
    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)

    #print(sp.pprint(derivatives.Z_omega, use_unicode=True))

    # print("Co-ordinates")
    # sp.pprint(sp.series(coords.R, coords.epsilon, 0, 4), use_unicode=True)
    # sp.pprint(sp.series(coords.Z, coords.epsilon, 0, 4), use_unicode=True)
    

    # print()
    # print("Metrics")

    sp.pprint(sp.simplify(sp.simplify(sp.series(j, coords.epsilon, 0, 2))), use_unicode=True)
    # sp.pprint(sp.series(metric.g_rw, coords.epsilon, 0, 2), use_unicode=True)
    #sp.pprint(sp.trigsimp(sp.simplify(sp.series(metric.g_ww, coords.epsilon, 0, 2))), use_unicode=True)