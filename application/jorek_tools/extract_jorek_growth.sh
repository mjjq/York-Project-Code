#!/bin/sh

# Execute this in the root directory of a simulation run

$JOREK_UTIL/extract_live_data.sh magnetic_growth_rates

mv magnetic_growth_rates.dat ./postproc
