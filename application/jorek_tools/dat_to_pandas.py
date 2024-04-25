import pandas as pd
import numpy as np
import sys
from typing import List

def get_parameter_names_from_dat(filename: str) -> List[str]:
	names = []
	with open(filename, 'r') as file:
		line = file.readline()
		# First term in names will be the comment hash, so use [1:] to ignore this
		names = line.split()[1:]

	return names


def dat_to_pandas(filename: str) -> pd.DataFrame:
	data = np.genfromtxt(filename)
	
	names = get_parameter_names_from_dat(filename)
	#print(names)
	#print(data)

	df = pd.DataFrame(data, columns=names)

	return df

if __name__=='__main__':
	fname = sys.argv[1]
	df = dat_to_pandas(fname)
	print(df)

	df.to_csv(fname.replace('.dat', '.csv'))
