from scipy.integrate import quad
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

from pyplot_helper import savefig

@np.vectorize
def Y(x):
    int_result, int_error = quad(
        lambda u: np.exp(-0.5*u*x**2)/(1-u**2)**(1/4),
        0,
        1.0
    )

    return -0.5 * x * int_result

if __name__=='__main__':
    xs = np.linspace(-100.0, 100.0, 10000)
    ys = Y(xs)

    y_func = UnivariateSpline(xs, ys, s=0)
    dydx = y_func.derivative()
    d2ydx2 = dydx.derivative()
    d3ydx3 = d2ydx2.derivative()

    M = xs * d3ydx3(xs)/d2ydx2(xs)
    #print(Y(0.0))

    fig, ax = plt.subplots(1, figsize=(4,3))
    #ax.plot(xs, y_func(xs), label='y')
    #ax.plot(xs, d2ydx2(xs), label='d2y/dx2')
    #ax.plot(xs, d3ydx3(xs), label='d3y/dx3')
    #ax.plot(xs, ys)

    ax.plot(xs, M, color='black', label="$XY(X)'''/Y(X)''$")
    ax.set_ylim(-4.0, -2.0)
    #ax.hlines(
        #-3.0, min(xs), max(xs), linestyle='--', color='red',
        #label='Asymptote (Y=-3)'
    #)

    ax.set_xlabel("X")
    ax.set_ylabel("$XY(X)'''/Y(X)''$")

    #ax.legend()
    ax.grid(which='major')

    fig.tight_layout()

    savefig("XY-plot")

    plt.show()
