import pandas as pd

from helpers import classFromArgs
from helpers import TimeDependentSolution

def constant_psi_approximation(ql_sol: TimeDependentSolution):
    d_delta_prime = ql_sol.delta_prime * ql_sol.w_t

def test_const_psi_on_data():
    m=2
    n=1
    S=1e8
    s=5.84863459819362
    r_s=0.7962252761034401

    fname = "./output/18-08-2023_16:41_new_ql_tm_time_evo_(m,n,A)=(2,1,1e-10).csv"
    df = pd.read_csv(fname)
    ql_sol = classFromArgs(TimeDependentSolution, df)

