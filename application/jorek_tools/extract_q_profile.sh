#!/bin/sh

./jorek2_postproc < $JOREK_TOOLS/get_flux.pp

cd postproc

python3 $JOREK_TOOLS/dat_to_pandas.py exprs_averaged_s00000.dat

python3 $JOREK_TOOLS/jorek_dat_to_array.py exprs_averaged_s00000.csv qprofile_s00000.dat
