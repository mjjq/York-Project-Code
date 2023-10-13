from dataclasses import fields, dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path
from zipfile import ZipFile
import json
from io import BytesIO
from typing import Tuple

from matplotlib import pyplot as plt
import pandas as pd

def savefile_fullpath(name: str):
    date_time = datetime.now().strftime("%d-%m-%Y_%H:%M")

    p = Path("./output")
    try:
        p.mkdir()
    except FileExistsError:
        print("Path exists. Skipping")

    return f"./output/{date_time}_{name}"


def savecsv(name: str, df: pd.DataFrame):
    """
    Save a pandas dataframe to csv. This function formats the filename with a
    date and time and stores it to an output folder. Note: The folder must
    be created to manually for this to work.
    """
    s = savefile_fullpath(name) + ".csv"
    print(f"Saving csv: {s}")

    df.to_csv(s, index=False)

def savefig(name: str, **kwargs):
    """
    Save the current pyplot frame as a .png. This function formats the filename
    with a date and time and stores it to an output folder. Note: The folder must
    be created to manually for this to work.
    """
    s = savefile_fullpath(name) + ".png"
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

def class_from_dict(className: dataclass, argDict: dict) -> dataclass:
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {k : v for k, v in argDict.items() if k in fieldSet}
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

@dataclass
class TearingModeParameters():
    # Poloidal number of the tearing mode
    poloidal_mode_number: int
    # Toroidal number of the tearing mode
    toroidal_mode_number: int
    # Ratio of resistive timescale to Alfven timescale
    lundquist_number: float
    # On-axis safety factor (normalised)
    axis_q: float
    # Magnetic shear at the resonant surface
    magnetic_shear: float
    # Minor radial co-ordinate of the resonant surface
    # (normalised to minor radius)
    resonant_surface: float

def dataclass_to_disk(name: str, cls: dataclass):
    """
    Save a dataclass to disk as a .csv file (automatically converts to a
    pandas DataFrame).
    """
    savecsv(name, pd.DataFrame(asdict(cls)))

def sim_to_disk(name: str,
                params: TearingModeParameters,
                time_dep_sol: TimeDependentSolution):
    params_df_bytes = json.dumps(asdict(params))
    time_dep_sol_df_bytes = pd.DataFrame(
        asdict(time_dep_sol)
    ).to_csv(index=False).encode('utf-8')

    s = savefile_fullpath(name) + ".zip"

    with ZipFile(s, "w") as zf:
        #param_name = "parameters.json"
        #zf.write(param_name)
        zf.writestr("parameters.json", params_df_bytes)

        #sol_name = "time_solution.csv"
        #zf.write(sol_name)
        zf.writestr("time_solution.csv", time_dep_sol_df_bytes)

def load_sim_from_disk(name: str) -> \
    Tuple[TearingModeParameters, TimeDependentSolution]:

    params_dataclass = None
    time_data_class = None

    with ZipFile(name, "r") as zf:
        params = zf.read("parameters.json")
        params_dataclass = class_from_dict(
            TearingModeParameters,
            json.loads(params)
        )

        time_data_str = BytesIO(zf.read("time_solution.csv"))
        time_data_class = classFromArgs(
            TimeDependentSolution,
            pd.read_csv(time_data_str)
        )

    return params_dataclass, time_data_class


def _test_to_disk_function():
    params = TearingModeParameters(
        2, 1, 1e8, 1.0, 5.2, 0.56
    )

    data = TimeDependentSolution(
        np.linspace(0.0, 1.0, 10),
        np.linspace(1.0, 2.0, 10),
        np.linspace(2.0, 3.0, 10),
        np.linspace(3.0, 4.0, 10),
        np.linspace(4.0, 5.0, 10),
        np.linspace(5.0, 6.0, 10)
    )

    sim_to_disk("test", params, data)

def _test_load_from_disk_function():
    params, sim_data = load_sim_from_disk(
        "./output/13-10-2023_12:12_test.zip"
    )

    print(params)

    print(sim_data)
    return

if __name__=='__main__':
    _test_load_from_disk_function()
