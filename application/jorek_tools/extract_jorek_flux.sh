#!/bin/sh

# Usage: Run from the root directory of a JOREK simulation.
# This script extracts psi(r_s) from JOREK's vtk file data and dumps it into a .csv
# file

rootdir=$(pwd)

cd ./vtk_no0_iplane1

fname="psi_t_data.csv"

if [ ! -f $fname ]; then
	python3 $JOREK_TOOLS/psi_t_from_vtk.py *.vtk
fi

cp $fname $rootdir/postproc

