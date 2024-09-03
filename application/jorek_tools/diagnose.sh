#!/bin/sh

cd postproc

gnuplot -persist -e 'filename="qprofile_s00000.dat"' $JOREK_TOOLS/plot_profile.gnuplot &
gnuplot -persist -e 'filename="exprs_averaged_s00000.dat"' $JOREK_TOOLS/plot_profile.gnuplot &
