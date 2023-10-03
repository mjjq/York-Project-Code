# Tearing mode solver

This repo contains Python implementations of different tearing mode solver
models. Below is a summary of what is inside each script.

## constant_psi_approximation.py

This script is deprecated and will be removed at a later date

## helpers.py

Contains some useful dataclass definitions and frequently used functions such
as figure/data saving/loading.

## linear_solver.py

Contains implementation of the outer tearing mode solution as well as some
functions to calculate the growth rate and discontinuity parameter (Delta')
in the linear regime.

## quasi_linear_solver.py

Early implementation of the quasi-linear time-dependent solver using the
`gamma' tearing mode model.

## new_ql_solver.py

Quasi-linear time-dependent solver using the `delta' tearing mode model.

## nl_td_solver.py

Time-dependent solver using the strongly non-linear Rutherford equation.

## non_linear_solver.py

Contains functions such as the generalised discontinuity parameter (Delta'(w)).

## phi_reconstruction.py

(Work in progress) script to reconstruct the full electric potential as a
function of x and t from the quasi-linear flux solution.

## solution_plotter.py

Various functions to load saved tearing mode solutions and analyse/plot them.

## time_dependent_solver.py

(Requires updating) Time-dependent solver using the linear theory model.

## validate_simplified_delta.py

Script which reconstructs full layer width as a function of (X, t) and
validates this against the approximate form of the quasi-linear layer width
via Delta'(X,t).

Also contains function to determine validity of the constant-psi approximation
from the delta model.

## y_sol.py

Contains integral solution used in the calculation of Delta' in the inner layer.
