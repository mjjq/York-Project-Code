#!/bin/bash

for d in $(find ./ -type d -name "run_*"); do (cd "$d" && $JOREK_UTIL/extract_live_data.sh -si magnetic_energies > magnetic_energies.dat); done
