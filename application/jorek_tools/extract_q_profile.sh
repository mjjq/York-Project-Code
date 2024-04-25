#!/bin/sh

./jorek2_postproc < $JOREK_TOOLS/get_flux.pp

python3 $JOREK_TOOLS/jorek_dat_to_array.py exprs_averaged_s00000.dat qprofile_s00000.dat
