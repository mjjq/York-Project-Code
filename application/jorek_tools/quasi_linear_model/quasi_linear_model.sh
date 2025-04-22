#!/bin/sh

# USAGE: cd to root directory of a jorek simulation then run this script
# This script uses postproc to extract the qprofile and current profile,
# mapping it from normalised psi to the radial co-ordinate r. This is used
# in model_jorek_params.py inside the experiments subfolder of the tearing
# mode solver code.

./jorek2_postproc < $JOREK_TOOLS/quasi_linear_model/get_flux.pp

cd postproc

python3 -m experiments.jorek_growth_comparison.model_jorek_params
