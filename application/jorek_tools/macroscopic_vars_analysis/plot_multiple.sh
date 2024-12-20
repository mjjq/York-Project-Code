#!/bin/sh

python3 $JOREK_TOOLS/macroscopic_vars_analysis/plot_quantities.py -f $(cat useful_runs.txt | sed 's/$/magnetic_energies.dat/') -ci 2 -ys log -x0 1 -yl 'Magnetic energy [JOREK] units' -o output.png
