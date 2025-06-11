#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm temp.dat
for i in $(seq 1 $2)
do
	$SCRIPT_DIR/dpsi_rs_time.sh $i 1 $1 >> temp.dat
done

gnuplot -e 'filename="temp.dat"' $SCRIPT_DIR/plot_dpsi.plg --persist
rm temp.dat
