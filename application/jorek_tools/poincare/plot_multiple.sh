#!/bin/bash

function help() {
	echo "Provide a list of restart .h5 files as arguments to this script"
}

function restart_number() {
	restart_filename=$1
	echo ${restart_filename%.h5} | grep -o '[0-9]\+'
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
__plq() {
	$(python3 $SCRIPT_DIR/../macroscopic_vars_analysis/plot_quantities.py "$@")
}

function gen_poincare() {
	restart_file=$1
	restart_no=$(restart_number $restart_file)
	poincare_rz_filename=$(echo poinc_R-Z_$restart_no.dat)
	poincare_rt_filename=$(echo poinc_rho-theta_$restart_no.dat)

	if [ ! -f $poincare_rz_filename ]; then
		mv jorek_restart.h5 jorek_restart.h5.old
		cp $restart_file jorek_restart.h5
		./jorek2_poincare < inmastu
		mv poinc_R-Z.dat $poincare_rz_filename
		mv poinc_rho-theta.dat $poincare_rt_filename
	fi
}

function plot_poincare() {
	restart_file=$1
	gen_poincare $restart_file
	restart_no=$(restart_number $restart_file)
	poincare_rz_filename=$(echo poinc_R-Z_$restart_no.dat)
	__plq -f $poincare_rz_filename -t . -fs 3.2 3.0 -xl "R (m)" -yl "Z (m)" -ms 0.2 -o poinc_$restart_no.png
}

function gen_poincare_multiple() {
	for var in "$@"
	do
		gen_poincare $var
	done
}

function plot_poincare_multiple() {
	for var in "$@"
	do
		plot_poincare $var
	done
}
