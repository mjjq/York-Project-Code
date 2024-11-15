import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
#from scipy.interpolate import CloughTocher2DInterpolator, UnivariateSpline # Takes ages to import
import os
import numpy as np
from typing import List, Tuple
import sys
import f90nml

import imports
from tearing_mode_solver.outer_region_solver import rational_surface, magnetic_shear
from tearing_mode_solver.delta_model_solver import nu, mode_width
from tearing_mode_solver.helpers import (
    classFromArgs,
    TimeDependentSolution,
    savefig,
    load_sim_from_disk,
    TearingModeParameters
)
from tearing_mode_solver.outer_region_solver import island_width
#from tearing_mode_solver.algebraic_fitting import get_parab_coefs # Takes ages to import
from jorek_tools.calc_jorek_growth import growth_rate, _name_time, _name_flux
# from jorek_tools.jorek_dat_to_array import q_and_j_from_csv
# from jorek_tools.psi_t_from_vtk import jorek_flux_at_q
# from jorek_tools.time_conversion import jorek_to_alfven_time