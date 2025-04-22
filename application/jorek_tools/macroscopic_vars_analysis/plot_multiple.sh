#!/bin/sh

python3 $JOREK_TOOLS/macroscopic_vars_analysis/plot_quantities.py -f $(cat useful_runs.txt | sed 's/$/\/magnetic_energies.dat/') -yi 2 -ys log -x0 0 -xl "Time (s)" -yl 'Magnetic energy [JOREK] units' -l $(cat labels.txt) -o mag_energies.png
python3 $JOREK_TOOLS/macroscopic_vars_analysis/plot_quantities.py -f $(cat useful_runs.txt | sed 's/$/\/magnetic_growth_rates.dat/') -yi 2 -x0 0 -xl "Time (s)" -yl "Magnetic growth rate (s$^{-1}$)" -l $(cat labels.txt) -o mag_growth_rates.png
