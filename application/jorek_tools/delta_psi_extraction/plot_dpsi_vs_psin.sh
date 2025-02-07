#!/bin/bash


function usage() {
        echo ""
        echo "Extract delta_psi as a function of Psi_N for a given set of poloidal modes."
        echo "Uses data extracted using postprocs Four2D method"
        echo ""
        echo "REMARK: Must launch from root of simulation output"
        echo ""
        echo "Usage: `basename $0` <time step> <poloidal_mode_number> <toroidal_mode_number>"
        echo ""
        echo ""
}

# --- Evaluate command line parameters
if [ $# -lt 1 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  usage
  exit;
fi

if [ $# -lt 3 ]; then
	usage
	exit;
fi

plot=0
for i in "$@"
do
	if [[ $i == "-p" ]]; then
		plot=1
		break
	fi
done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

timestep=$(printf "%05d" $1)
# TODO: Add option to specify n
mraw=$2
m=$(printf "%03d" $mraw)
nraw=$3
n=$(printf "%03d" $nraw)

filename="postproc/exprs_four2d_s`echo $timestep`_absolute_value_n`echo $n`.dat"

ret=$(cat $filename | grep -A 100 "m/n=+$m" | sed "s/# absolute_values.*/\"m=$mraw, n=$nraw\"/")

if [[ $plot == 1 ]]; then
	echo "$ret" > modes.dat.tmp
	gnuplot -e 'filename="modes.dat.tmp"' $SCRIPT_DIR/plot_dpsi_psin.plg
	rm modes.dat.tmp
else
	echo "$ret"
fi
