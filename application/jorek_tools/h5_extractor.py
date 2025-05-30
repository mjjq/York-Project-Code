from typing import Tuple, List
import h5py
from dataclasses import dataclass

@dataclass
class JOREKTimeData:
    t_steps: List[int]
    t_vals: List[float]
    t_si_vals: List[float]

def get_si_time_at_tstep(h5_data: h5py.File) -> Tuple[int, float, float]:
    t_step = int(h5_data['index_now'][0])
    t_now = float(h5_data['t_now'][0])
    t_norm = float(h5_data['t_norm'][0])

    return t_step, t_now, t_now*t_norm

def get_si_timesteps(h5_filenames: List[str]) -> JOREKTimeData:
    list_of_tuples = [get_si_time_at_tstep(
        h5py.File(filename, mode='r')
    ) for filename in h5_filenames]

    tsteps, t_vals, t_si_vals = list(zip(*list_of_tuples))

    return JOREKTimeData(
        t_steps=list(tsteps),
        t_vals=list(t_vals),
        t_si_vals=list(t_si_vals)
    )



if __name__=='__main__':
    import sys

    filenames = sys.argv[1:]

    print(get_si_timesteps(filenames))

