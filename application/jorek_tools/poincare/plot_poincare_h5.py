import h5py
import matplotlib.pyplot as plt


filename = "poincare.h5"

f = h5py.File(filename, 'r')

R = f['r']
Z = f['z']

plt.figure('poincare')
plt.scatter(R, Z, s=0.1, color='black')
plt.show()
