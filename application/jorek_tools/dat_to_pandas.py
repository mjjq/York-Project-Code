import pandas as pd
import numpy as np
import sys
from typing import List

from debug.log import logger

def get_parameter_names_from_dat(filename: str) -> List[str]:
	names = []
	with open(filename, 'r') as file:
		line = file.readline().replace('"', '')
		# First term in names will be the comment hash, so use [1:] to ignore this
		names = line.split()

	return names


def dat_to_pandas(filename: str) -> pd.DataFrame:
	data = np.genfromtxt(filename)
	
	names = get_parameter_names_from_dat(filename)
	#print(names)
	#print(data)
	num_columns = data.shape[1]
	if len(names) == num_columns + 1:
		logger.warning(f"{num_columns} Columns and {len(names)} column names")
		logger.warning("Removing first name (likely hash delimiter)")
		names = names[1:]
	elif (len(names) > num_columns+1) or (len(names) < num_columns):
		raise ValueError(
			f"Too many/not enough column names ({len(names)} "
			f"names, {num_columns} cols). "
			 "Check your dat file"
		)

	df = pd.DataFrame(data, columns=names).dropna(how='all')

	return df

if __name__=='__main__':
	fname = sys.argv[1]
	df = dat_to_pandas(fname)
	print(df)

	df.to_csv(fname.replace('.dat', '.csv'))
