#!/bin/bash

function usage() {
	echo ""
	echo "Extract delta_psi(r_s) as a function of time for a given mode."
	echo "Uses data extracted using postprocs Four2D method"
	echo "Note: working directory must be in the postproc folder where"
	echo "Four2D output is stored."
	echo ""
	echo "REMARK: Must launch from root of simulation output"
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

# m/n need leading zeros to match 3 digit number format of file
m=$(printf "%03d" $1)
n=$(printf "%03d" $2)
rs=$3
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

function extract() {
	grep -r "m/n=+$m" -A 100 *absolute_value_n$n* | awk -v rs=$rs '{ if ($2 > rs-0.005 && $2 < rs+0.005  ) print $1 " " $2 " " $3 }' | sed "s/exprs_four2d_s//" | sed "s/_absolute_value_n$n.dat-//"
}

function extract_with_si_time() {
	timemap="$($SCRIPT_DIR/time.sh log)"
	cd postproc
	extracted="$(extract)"
	join <(echo "$extracted") <(echo "$timemap")
	cd ..
}

plot=0

for var in "$@"
do
	if [ "$var" == "-p" ]; then
		plot=1
	fi
done

if [ $plot == 1 ]; then
	outfile="dpsi_temp.dat"
	extract_with_si_time>$outfile
	echo $outfile
	gnuplot -e "filename='$(echo $outfile)'" $SCRIPT_DIR/plot_dpsi.plg --persist
	rm "$outfile"
else
	extract_with_si_time
fi
