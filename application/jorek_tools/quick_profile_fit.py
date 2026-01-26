import numpy as np
from dataclasses import dataclass
from typing import Tuple
from functools import partial
from scipy.optimize import curve_fit

def ff_polynomial(r: float, 
                  c1: float, 
                  c2: float, 
                  c3: float) -> float:
    return 1.0 + c1*r + c2*r**2 + c3*r**3

def ff_tanh(r: float,
            c4: float,
            c5: float) -> float:
    return 0.5 * (1-np.tanh((r-c5)/c4))

def ff_profile(r: float,
               c1,c2,c3,c4,c5: float,
               ff0: float,
               ff1: float) -> float:
    return (ff0-ff1)*ff_polynomial(r, c1, c2, c3) * ff_tanh(r, c4, c5) + ff1

def test_ff_vals():
    r_vals = np.linspace(0.0, 1.0, 100)
    ff_vals = ff_profile(
        r_vals,
        9.52738538, -14.42541833,  53.43849888,  0.20897478,  0.24732681,
        -0.357,
        0.0
    )

    return r_vals, ff_vals

def plot_ff_profile_test():
    r_vals, ff_vals = test_ff_vals()

    from matplotlib import pyplot as plt
    plt.plot(r_vals, ff_vals)
    plt.show()

def fit_ffprime_profile(r_vals: np.array,
                        ff_vals: np.array):
    ff_partial = partial(ff_profile, ff0=ff_vals[0], ff1=ff_vals[-1])

    popt, pcov = curve_fit(ff_partial, r_vals, ff_vals)

    print(popt)
    ff_vals_fit = ff_partial(r_vals, *popt)
    
    from matplotlib import pyplot as plt
    plt.plot(r_vals, ff_vals)
    plt.plot(r_vals, ff_vals_fit)
    plt.plot(*test_ff_vals())
    plt.show()

def load_ff_vals_from_file(fname: str) -> Tuple[np.array, np.array]:
    data = np.loadtxt(fname)
    r_vals = data[:,0]
    ff_vals = data[:,1]

    return r_vals, ff_vals

def apply_tanh_cutoff(r_vals: np.array,
                      ff_vals: np.array) -> np.array:
    tanh_cutoff_vals = ff_tanh(r_vals, 0.025, 0.85)
    return ff_vals * tanh_cutoff_vals

if __name__=='__main__':
    import sys
    fname = sys.argv[1]

    r_vals, ff_vals = load_ff_vals_from_file(fname)
    #fit_ffprime_profile(r_vals, ff_vals)

    ff_vals_new = apply_tanh_cutoff(r_vals, ff_vals)

    print("\n".join([f"{r:.10f} {ff:.10f}" for r,ff in zip(r_vals, ff_vals_new)]))
