#!/bin/sh

function usage() {
	echo ""
	echo "Extract delta_psi(r_s) as a function of time for a given mode."
	echo "Uses data extracted using postprocs Four2D method"
	echo "Note: working directory must be in the postproc folder where"
	echo "Four2D output is stored."
	echo ""
	echo "Usage: `basename $0` <poloidal mode number> <toroidal mode number> <rs> [options]"
	echo ""
	echo "	-p Plot the output instead of directing to STDOUT"
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

m=$1
n=$2
rs=$3

function extract() {
	grep -r "m/n=+00$m" -A 100 *absolute_value_n00$n* | awk -v rs=$rs '{ if ($2 > rs-0.005 && $2 < rs+0.005  ) print $1 " " $2 " " $3 }' | sed "s/exprs_four2d_s//" | sed "s/_absolute_value_n00$n.dat-//"
}

plot=0

for var in "$@"
do
	if [ "$var" == "-p" ]; then
		plot=1
	fi
done

if [ $plot == 1 ]; then
	outfile=dpsi_vs_time_m"$m"_n"$n"_rs"$rs".dat
	extract>$outfile
	gnuplotcmd="plot '$(echo $outfile)' using 1:3 with lines"
	SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
	gnuplot -e "filename='$(echo $outfile)'" $SCRIPT_DIR/plot_dpsi.plg --persist
else
	extract
fi
