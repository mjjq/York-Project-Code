#!/bin/sh

cd postproc

mkdir -p ./output

# Get the latest model simulation from output directory
modelfname=$(ls ./output/ -Art | grep .zip | tail -n 1)

if [[ $modelfname ]]
then
	echo "Found $modelfname"
	modelfname=./output/$modelfname
fi

# JOREK namelist file
nmlfname="../intear"

# JOREK psi(r_s, t) data in csv format, should have same filename as always
jorekfname="psi_t_data.csv"

# q profile data in .dat format, extracted directly from jorek2postproc
qprofname="qprofile_s00000.dat"

# Current profile and psi->r mapping data in .csv format. Initially extracted from
# postproc as a .dat file then converted to .csv
currprofname="exprs_averaged_s00000.csv"

# Magnetic energy data from JOREK in .csv format.
jorekmaggrowth="magnetic_growth_rates.csv"

echo $modelfname
python3 $JOREK_TOOLS/../experiments/jorek_growth_comparison/compare_flux.py \
	$nmlfname \
	$qprofname \
	$currprofname \
	$jorekfname \
	$modelfname

# python3 $JOREK_TOOLS/../experiments/jorek_growth_comparison/compare_model_growth_to_jorek.py $modelfname $jorekmaggrowth
