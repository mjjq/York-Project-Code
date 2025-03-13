#!/bin/bash

function usage() {
	echo ""
	echo "Extract magnetic island width as a function of time for a given mode."
	echo "Uses data extracted using postprocs Four2D method and a custom magnetic"
	echo "shear postproc expression. Uses w=sqrt(m/n*delta_psi"
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

if [ $# -lt 2 ]; then
	usage
	exit;
fi

# m/n need leading zeros to match 3 digit number format of file
mraw=$1
nraw=$2
m=$(printf "%03d" $mraw)
n=$(printf "%03d" $nraw)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
mn=$(echo "$mraw/$nraw" | bc)

function extract_shear() {
	grep -r "" shear_s00000*.dat | awk -v mn=$mn '{ if ($4 > mn-0.005 && $4 < mn+0.005) print $1 " " $5}'
}

function extract_rs() {
	grep -r "" shear_s00000*.dat | awk -v mn=$mn '{ if ($4 > mn-0.005 && $4 < mn+0.005) print $1 " " $3}'
}

function extract_delta_psi() {
	rsvals="$(extract_rs)"
	fourierfiles="$(ls *absolute_value_n$n*)"
	for file in ${fourierfiles}
	do
		timestamp=$(echo $file | sed "s/exprs_four2d_s//" | sed "s/_absolute_value_n$n.dat//")
		rs=$(echo "$rsvals" | grep $timestamp | awk '{print $2}')
		grep -r "m/n=+$m" -A 100 $file | awk -v rs=$rs -v timestamp=$timestamp '{ if ($1 > rs-0.005 && $1 < rs+0.005  ) print timestamp " " $2 }'
	done
}

function extract_island_width() {
	cd postproc
	sheartime="$(extract_shear)"
	deltapsi="$(extract_delta_psi)"
	join <(echo "$sheartime") <(echo "$deltapsi") | awk -v mn=$mn '{print $1 " " 4.0*sqrt(mn*$3/$2)}'
	cd ..
}

function extract_with_si_time() {
	timemap="$(sort -k1 <($SCRIPT_DIR/time.sh log*))"
	extracted="$(extract_island_width)"
	echo "\"m=$mraw, n=$nraw\""
	join <(echo "$extracted") <(echo "$timemap")
	echo ""
	echo ""
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
#	rm "$outfile"
else
	extract_with_si_time
fi
