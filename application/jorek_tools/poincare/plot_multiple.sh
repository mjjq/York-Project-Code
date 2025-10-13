#!/bin/bash

function help() {
	echo "Provide a list of restart .h5 files as arguments to this script"
}

function restart_number() {
	restart_filename=$1
	echo ${restart_filename%.h5} | grep -o '[0-9]\+'
}

_poincare_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
__plq() {
	$(python3 $_poincare_SCRIPT_DIR/../macroscopic_vars_analysis/plot_quantities.py "$@")
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
		mv jorek_restart.h5.old jorek_restart.h5
	fi
}

function plot_poincare() {
	restart_file=$1
	gen_poincare $restart_file
	restart_no=$(restart_number $restart_file)
	poincare_rz_filename=$(echo poinc_R-Z_$restart_no.dat)
	__plq -f $poincare_rz_filename -t o -xl "R (m)" -yl "Z (m)" -ms 0.25 -a equal -o poinc_R-Z_$restart_no.png

	poincare_rt_filename=$(echo poinc_rho-theta_$restart_no.dat) 
	__plq -f $poincare_rt_filename -t o -xl "$\rho/a$" -yl "$\theta$ (rad)" -fs 4 4 -ms 0.25 -o poinc_rho-theta_$restart_no.png 
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
