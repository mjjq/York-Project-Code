import numpy as np
from io import StringIO
from argparse import ArgumentParser
import re

def read_delta_gw_raw(filename: str) -> np.array:
    with open(filename, 'r') as f:
        d = f.read()
        
    # Ignore headers and footers
    lines = d.split("\n")[4:-4]

    # Ignore row headers
    lines_no_row_header = [line[4:] for line in lines]
   
    raw_matrix_str = "\n".join(lines_no_row_header)

    raw_matrix = np.loadtxt(StringIO(raw_matrix_str))

    return raw_matrix

def get_left_real_matrix(delta_gw_raw: np.array) -> np.array:
    return delta_gw_raw[::2,::4]

def get_right_real_matrix(delta_gw_raw: np.array) -> np.array:
    return delta_gw_raw[1::2,2::4]

def get_left_imag_matrix(delta_gw_raw: np.array) -> np.array:
    return delta_gw_raw[::2,1::4]

def get_right_imag_matrix(delta_gw_raw: np.array) -> np.array:
    return delta_gw_raw[1::2,3::4]

def delta_prime(delta_gw_raw: np.array, surface_index: int) -> float:
    """
    Delta' for a specific surface index (this index does not
     necessarily) correspond to the value of the rational surface

    :param: delta_gw_raw: 2x2 delta_gw matrix as read by
        read_delta_gw_raw()
    :param surface_index: The index of the rational surface in
        rdcon
    """
    #delta_gw_real_l = get_left_real_matrix(delta_gw_raw)
    #delta_gw_real_r = get_right_real_matrix(delta_gw_raw)

    array_index = surface_index-1

    delta_re_nlnl = delta_gw_raw[array_index, array_index]
    delta_re_nrnr = delta_gw_raw[array_index+1, array_index+2]
    delta_re_nlnr = delta_gw_raw[array_index, array_index+2]
    delta_re_nrnl = delta_gw_raw[array_index+1, array_index]

    #print(delta_re_nlnl, delta_re_nrnr, delta_re_nlnr, delta_re_nrnl)

    return delta_re_nlnl+delta_re_nrnr - (delta_re_nrnl+delta_re_nlnr)


def time_from_g_filename(filename: str) -> float:
    """
    Extract time from geqdsk filename
    """
    return float(re.findall(r'[+-]?\d+\.\d+(?:[Ee][+-]?\d+)?', filename)[0])
    

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "filenames", nargs="+", type=str, 
        help="List of delta_gw.out ascii files"
    )
    parser.add_argument(
        "-s", "--surfaces", nargs="+", type=int, 
        help="List of singular surfaces to evaluate", default=[1]
    )
    parser.add_argument(
        "-t", "--print-times", action='store_true',
        help="Enable this flag to print the time alongside the delta value"
    )
    args = parser.parse_args()

    filenames = args.filenames
    surface_indices = args.surfaces
    #dgw_raw = read_delta_gw_raw(filename)

    for filename in filenames:
        dgw_raw = read_delta_gw_raw(filename)
        for surf in surface_indices:
            dp = delta_prime(dgw_raw, surf)
            if args.print_times:
                time = time_from_g_filename(filename)
                print(time, dp)
            else:
                print(dp)


