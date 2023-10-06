from dataclasses import fields, dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd



def savecsv(name: str, df: pd.DataFrame):
    """
    Save a pandas dataframe to csv. This function formats the filename with a
    date and time and stores it to an output folder. Note: The folder must
    be created to manually for this to work.
    """
    date_time = datetime.now().strftime("%d-%m-%Y_%H:%M")
    s = f"./output/{date_time}_{name}.csv"
    print(f"Saving csv: {s}")

    df.to_csv(s, index=False)

def savefig(name: str, **kwargs):
    """
    Save the current pyplot frame as a .png. This function formats the filename
    with a date and time and stores it to an output folder. Note: The folder must
    be created to manually for this to work.
    """
    date_time = datetime.now().strftime("%d-%m-%Y_%H:%M")

    p = Path("./output")
    try:
        p.mkdir()
    except FileExistsError:
        print("Path exists. Skipping")

    s = f"./output/{date_time}_{name}.png"
    print(f"Saving figure: {s}")
    plt.savefig(s, dpi=300)

def classFromArgs(className: dataclass, df: pd.DataFrame) -> dataclass:
    """
    Convert a pandas DataFrame to a dataclass. The dataclass is given by
    className and its fields must match the columns in df.
    """
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {col : np.array(df[col]) for col in df.columns
                       if col in fieldSet}
    return className(**filteredArgDict)

@dataclass
class TimeDependentSolution():
    """
    Generic class containing numerical solution to time-dependent flux evolution
    equation. This can be generated either by the strongly non-linear time-
    dependent solver or either of the quasi-linear solvers.
    """
    # Array of times pertaining to the perturbed flux solution.
    times: np.array
    # The perturbed flux solution as an array of floats. Each element has a
    # corresponding time given in times.
    psi_t: np.array
    # First derivative of the pertured flux with respect to time.
    dpsi_dt: np.array
    # Second derivative of the perturbed flux with respect to time.
    d2psi_dt2: np.array
    # Magnetic island widths associated with each element in psi_t.
    w_t: np.array
    # Discontinuity parameters associated with each element in psi_t.
    delta_primes: np.array


def dataclass_to_disk(name: str, cls: dataclass):
    """
    Save a dataclass to disk as a .csv file (automatically converts to a
    pandas DataFrame).
    """
    savecsv(name, pd.DataFrame(asdict(cls)))

if __name__=='__main__':
    #savefig("this_is_a_test")
    #classFromArgs(float, 1.0)
    pass
