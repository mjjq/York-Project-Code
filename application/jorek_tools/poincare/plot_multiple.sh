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
	poincare_rz_filename=$(echo poinc_R-Z_$restart_no.txt)
	poincare_rt_filename=$(echo poinc_rho-theta_$restart_no.txt)

	if [ ! -f $poincare_rz_filename ]; then
		tmp_folder=tmp_$restart_no
		mkdir $tmp_folder
		cd $tmp_folder
		ln -s ../* .
		rm jorek_restart.h5

		cp ../$restart_file jorek_restart.h5
		./jorek2_poincare < inmastu
		mv poinc_R-Z.dat ../$poincare_rz_filename
		mv poinc_rho-theta.dat ../$poincare_rt_filename

		cd ../
		rm -r $tmp_folder
	fi
}

function plot_poincare() {
	python3 -m jorek_tools.poincare.plot_animation "$@"
}

function gen_poincare_multiple() {
	for var in "$@"
	do
		gen_poincare $var
	done
}


function gen_poincare_parallel() {
	export -f gen_poincare
	export -f restart_number
	ls jorek[0-9]*.h5 | xargs -t -P $1 -I {} bash -c 'gen_poincare "{}"'
}
