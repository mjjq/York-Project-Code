#!/bin/sh

cd postproc

python3 $JOREK_TOOLS/../experiments/jorek_growth_comparison/model_jorek_params.py exprs_averaged_s00000.csv qprofile_s00000.dat
