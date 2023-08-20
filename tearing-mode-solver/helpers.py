from dataclasses import fields, dataclass
from datetime import datetime
import numpy as np

from matplotlib import pyplot as plt
import pandas as pd

def savecsv(name: str, df: pd.DataFrame):
    date_time = datetime.now().strftime("%d-%m-%Y_%H:%M")
    s = f"./output/{date_time}_{name}.csv"
    print(f"Saving csv: {s}")

    df.to_csv(s, index=False)

def savefig(name: str, **kwargs):
    date_time = datetime.now().strftime("%d-%m-%Y_%H:%M")

    s = f"./output/{date_time}_{name}.png"
    print(f"Saving figure: {s}")
    plt.savefig(s, dpi=300)

def classFromArgs(className, df):
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {col : np.array(df[col]) for col in df.columns
                       if col in fieldSet}
    return className(**filteredArgDict)

@dataclass
class TimeDependentSolution():
    times: np.array
    psi_t: np.array
    dpsi_dt: np.array
    d2psi_dt2: np.array
    w_t: np.array
    delta_primes: np.array

if __name__=='__main__':
    savefig("this_is_a_test")
