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
	
	use_fmhd_flag=$2
	use_fmhd=false
	
	nprocs_flag=$3
	nprocs=1

	extension="txt"

	if [[ $use_fmhd_flag = "-f" ]]; then
		use_fmhd=true
		extension="vtk"
	fi

	if [[ $nprocs_flag ]]; then
		nprocs=$nprocs_flag
	fi

	poincare_rz_filename=$(echo poinc_R-Z_$restart_no.$extension)
	poincare_rt_filename=$(echo poinc_rho-theta_$restart_no.$extension)

	if [ ! -f $poincare_rz_filename ]; then
		tmp_folder=tmp_$restart_no
		mkdir $tmp_folder
		cd $tmp_folder
		ln -s ../* .
		rm jorek_restart.h5

		cp ../$restart_file jorek_restart.h5
		
		if [[ $use_fmhd = false ]]; then
			./jorek2_poincare < inmastu
			mv poinc_R-Z.dat ../$poincare_rz_filename
			mv poinc_rho-theta.dat ../$poincare_rt_filename
		else
			rm connection.vtk
			mpirun -np $nprocs ./jorek2_connection_fmhd < inmastu
			mv connection.vtk ../$poincare_rz_filename
		fi

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
	every_nth_file=$2
	if [[ ! "$every_nth_file" ]]; then
		every_nth_file=1
	fi

	modulo=1
	if [[ "$every_nth_file" == 1 ]]; then
		modulo=0
	fi

	printf '%s\n' jorek[0-9]*.h5 | awk -v n="$every_nth_file" -v m="$modulo" 'NR % n == m' | xargs -t -P $1 -I {} bash -c 'gen_poincare "{}"'
}

function gen_poincare_parallel_fmhd() {
	export -f gen_poincare
	export -f restart_number
	every_nth_file=$2
	if [[ ! "$every_nth_file" ]]; then
		every_nth_file=1
	fi

	modulo=1
	if [[ "$every_nth_file" == 1 ]]; then
		modulo=0
	fi

	for f in $(printf '%s\n' jorek[0-9]*.h5 | awk -v n="$every_nth_file" -v m="$modulo" 'NR % n == m'); do
		gen_poincare $f -f $1
	done
}
