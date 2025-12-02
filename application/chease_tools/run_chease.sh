#!/bin/bash

run_chease() {
	./prof < prof_namelist |& tee prof_output.out
	cp EXPEQ_INIT EXPEQ
	./chease < chease_namelist |& tee chease_output.out
}

chease_to_jorek() {
	echo "Generating profiles..."

	./eqdsk2jorek < EQDSK_COCOS_02.OUT

	mv jorek_temperature jorek_temperature_orig
	central_t=$(head -n 1 jorek_temperature_orig | awk '{print $2}')
	echo $central_t
	cat jorek_temperature_orig | awk -v t0="$central_t" '{print $1 " " ($2 < 1e-6*t0 ? 1e-6*t0 : $2)}' > jorek_temperature

	mv jorek_density jorek_density_1f
	awk '{print $1 " " 0.5*$2}' jorek_density_1f > jorek_density_2f

	./o.chease_to_cols chease_output.out chease_cols.out

	#source $CHEASE_TOOLS/aliases.sh
	#chease_temperature chease_cols.out > jorek_temperature
	#chease_ffprime chease_cols.out > jorek_ffprime
	#chease_cols_psin_convert chease_cols.out > chease_cols_psin.out
}

function link_xtor_input() {
	chease_folder=$1

	if [ -d "$chease_folder" ]; then
		ln -s $chease_folder/ALL_PROFILES $chease_folder/EXPEQ $chease_folder/OUTXTOR $chease_folder/fort.8 $chease_folder/chease.dat $chease_folder/chease.bin .
	else
		echo "Folder doesn't exist, exiting."
	fi
}

function link_jorek_input() {
	chease_folder=$1

	if [ -d "$chease_folder" ]; then
		ln -s $chease_folder/jorek_ffprime $chease_folder/jorek_density_*f $chease_folder/jorek_temperature $chease_folder/inmastu .
	else
		echo "Folder doesn't exist, exiting."
	fi
}

function link_chease_files() {
	i=0
	for path in $@; do
		ln -s $path .
		bname="$(basename $path)"
		rename $bname chease_$i $bname
		((i++))
	done
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    run_chease
    chease_to_jorek
fi
