from dataclasses import dataclass
import sympy as sp
import pickle

@dataclass
class ToroidalCoordinates:
    R: sp.Function # Major radius
    Z: sp.Function # Vertical distance

    R0: sp.Symbol # Major radius of magnetic axis
    
    # New co-ords
    r: sp.Symbol # Minor radius
    omega: sp.Symbol # Generalised angle

    # Expansion params
    epsilon: sp.Symbol # Inverse aspect ratio tag
    ellip: sp.Symbol # Ellipticity tag

@dataclass
class EPFLShaping:
    D: sp.Function # Shafranov shift Delta(r)

    # Shaping
    S: sp.Function # Shaping coefficients (S_m(r))
    m: sp.Symbol # Shaping coef indices
    N: sp.Symbol # sum_{m=1}^N...

    P: sp.Function # P(r)

@dataclass
class LutjensShaping:
    D: sp.Function # Shafranov shift Delta(r)
    E: sp.Function # ellipticity E(r)
    T: sp.Function # Triangularity T(r)
    P: sp.Function # P(r)

    D_rr: sp.Function = None # d2D/dr2, derived later
    E_rr: sp.Function = None # d2E/dr2, derived later
    T_rr: sp.Function = None # d2T/dr2, derived later

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
    # Equation A.3. in Lutjens 1992 expanded
    # Given as expressions which are implicitly
    # equal to zero
    gs_equil:  sp.Expr # cos(0w) component at O(1)
    gs_1omega: sp.Expr # cos(1w) component at O(eps)
    gs_2omega: sp.Expr # cos(2w) component at O(e)
    gs_3omega: sp.Expr # cos(3w) component at O(eps)

    f: sp.Function # dpsi/dr = R0 B0 eps * f(r)
    g: sp.Function # G = R0 B0 g(r), g(r) = 1+O(eps^2)

    p: sp.Function # pressure p(r)

    B0: sp.Symbol # On-axis toroidal field


@dataclass
class State():
    coords: ToroidalCoordinates
    shaping: EPFLShaping | LutjensShaping
    derivatives: Derivatives
    metric: MetricTensor
    gs: GradShafVariables
    jacobian: sp.Expr

    def to_disk(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_disk(cls, filename: str):
        with open(filename, "rb") as f:
            state = pickle.load(f)

            assert isinstance(state, State)

            assert isinstance(
                state.coords, ToroidalCoordinates
            )
            assert isinstance(
                state.shaping, (EPFLShaping, LutjensShaping)
            )
            assert isinstance(
                state.derivatives, Derivatives
            )
            assert isinstance(
                state.metric, MetricTensor
            )
            assert isinstance(
                state.gs, GradShafVariables
            )
            assert isinstance(
                state.jacobian, sp.Expr
            )

            return state