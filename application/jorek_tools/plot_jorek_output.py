from dat_to_pandas import dat_to_pandas
import sys
from matplotlib import pyplot as plt

if __name__=='__main__':
	fname = sys.argv[1]
	df = dat_to_pandas(fname)

	df.plot(x='r_minor', y='zj')

	plt.show()
