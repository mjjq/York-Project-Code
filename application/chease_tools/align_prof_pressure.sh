#!/bin/bash

get_chease_beta() {
	chease_out_filename=$1

	grep "BETA-AXIS=" $chease_out_filename | awk '{ printf("%.10f\n", $NF) }'
}

get_prof_beta() {
	prof_out_filename=$1
	
	# Divide by 100 to convert from percentage to abs value
	grep -m 1 "beta total (%)" $prof_out_filename | awk '{ printf("%.10f\n", $NF/100) }'
}

approx_equal() {
	val=$1
	cmp=$2
	tolerance="0.001"

	echo "sqrt(($val/$cmp-1.0)^2) < $tolerance" | bc -l
}

betas_approx_equal() {
	chease_out_filename=$1
	prof_out_filename=$2

	chease_beta=$(get_chease_beta $chease_out_filename)
	prof_beta=$(get_prof_beta $prof_out_filename)

	approx_equal $chease_beta $prof_beta
}

get_cpress() {
	prof_namelist_filename=$1

	grep -a cpress $prof_namelist_filename | grep -v "!" | tail -1 | grep -Eo '[+-]?[0-9]+([.][0-9]+)?'
}

calc_new_cpress() {
	chease_out_filename=$1
	prof_out_filename=$2
	prof_namelist_filename=$3

	chease_beta=$(get_chease_beta $chease_out_filename)
	prof_beta=$(get_prof_beta $prof_out_filename)

	correction_factor=$(echo "$prof_beta $chease_beta" | awk '{printf "%f\n", $1/$2}')
	
	old_cpress=$(get_cpress $prof_namelist_filename)

	echo "$old_cpress $correction_factor" | awk '{printf "%f\n", $1*$2}'
}

update_cpress() {
	prof_namelist_filename=$1
	new_cpress=$2

	sed "s/.*cpress.*/cpress=$new_cpress,/" $prof_namelist_filename > tmp
	mv tmp $prof_namelist_filename
}

run_prof_chease() {
	./prof < prof_namelist > prof_output.out
	cp EXPEQ_INIT EXPEQ
	./chease < chease_namelist > chease_output.out
}

align_prof_pressure() {
	new_cpress=$(calc_new_cpress chease_output.out prof_output.out prof_namelist)
	update_cpress prof_namelist $new_cpress
}


loop_prof_pressure() {
	run_prof_chease

	cp prof_namelist prof_namelist.bak

	until [ $(betas_approx_equal chease_output.out prof_output.out) -eq 1 ]
	do
		echo "Betas not equal"
		echo "prof beta: " $(get_prof_beta prof_output.out)
		echo "chease beta: " $(get_chease_beta chease_output.out)

		align_prof_pressure
		run_prof_chease
	done

	echo "Betas close enough. Exiting"
}
