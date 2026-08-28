#!/bin/bash

function build_jorek() {
	mkdir -p ./build
	rm -r ./build/*

	rm jorek_model*

	make clean && make -j 32
	make -j 32 jorek2_postproc
	make -j 32 jorek2vtk
	make -j 32 jorek2_poincare
	make -j 32 jorek2_connection_fmhd

	mv jorek_model* ./build/jorek_model
	mv jorek2_postproc jorek2vtk jorek2_poincare jorek2_connection_fmhd ./build

	make clean && make -j 32 DEBUG=1
	mv jorek_model* ./build/jorek_model_debug

	./util/config.sh > ./build/settings.txt

	git log -1 > ./build/git_info.txt
	git branch >> ./build/git_info.txt
}

function gen_nperiod_scan(){
	for i in $(seq $1 $2); do
		./util/config.sh n_period=$i
		build_jorek
		mv build np$i
	done
}
