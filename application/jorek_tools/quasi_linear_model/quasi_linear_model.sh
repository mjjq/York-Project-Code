#!/bin/sh

# USAGE: cd to root directory of a jorek simulation then run this script
# This script uses postproc to extract the qprofile and current profile,
# mapping it from normalised psi to the radial co-ordinate r. This is used
# in model_jorek_params.py inside the experiments subfolder of the tearing
# mode solver code.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
./jorek2_postproc < $SCRIPT_DIR/get_flux.pp

cd postproc

python3 -m application.experiments.jorek_growth_comparison.model_jorek_params
