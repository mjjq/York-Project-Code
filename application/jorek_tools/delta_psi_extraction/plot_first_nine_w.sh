#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm temp.dat
for i in $(seq 2 4)
do
	echo "extracting m=$i n=1 data"
	$SCRIPT_DIR/island_width_extraction.sh $i 1 >> temp.dat
done

gnuplot -e 'filename="temp.dat"' $SCRIPT_DIR/plot_w_time.plg --persist
