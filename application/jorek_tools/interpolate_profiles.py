import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = input_file + "_interp5x"
resolution_factor = 5

data = np.loadtxt(input_file)
x = data[:, 0]
y = data[:, 1]

x_new = np.linspace(x[0], x[-1], len(x) * resolution_factor)
f = interp1d(x, y, kind='cubic')
y_new = f(x_new)
y_new = savgol_filter(y_new, window_length=11, polyorder=3)

np.savetxt(output_file, np.column_stack([x_new, y_new]), fmt="%.12e %.8e")
print(f"Saved {len(x_new)} points to {output_file}")
