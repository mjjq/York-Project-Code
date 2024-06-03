#!/bin/sh

# USAGE: cd to root directory of a jorek simulation then run this script
# This script uses postproc to extract the qprofile and current profile,
# mapping it from normalised psi to the radial co-ordinate r. This is used
# in model_jorek_params.py inside the experiments subfolder of the tearing
# mode solver code.

./jorek2_postproc < $JOREK_TOOLS/get_flux.pp

cd postproc

python3 $JOREK_TOOLS/dat_to_pandas.py exprs_averaged_s00000.dat

python3 $JOREK_TOOLS/jorek_dat_to_array.py exprs_averaged_s00000.csv qprofile_s00000.dat
