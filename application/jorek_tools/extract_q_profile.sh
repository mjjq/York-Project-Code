#!/bin/sh

./jorek2_postproc < $JOREK_ANALYSIS/extraction_scripts/get_flux.pp

python3 $JOREK_ANALYSIS/extraction_scripts/jorek_dat_to_array.py exprs_averaged_s00000.dat qprofile_s00000.dat
