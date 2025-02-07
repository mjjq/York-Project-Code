#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm temp.dat
timestep=$1
for i in $(seq 1 9)
do
	($SCRIPT_DIR/plot_dpsi_vs_psin.sh $timestep $i 1; printf "\n\n") >> temp.dat
done

gnuplot -e 'filename="temp.dat"' $SCRIPT_DIR/plot_dpsi_psin.plg --persist
#cat temp.dat
rm temp.dat
