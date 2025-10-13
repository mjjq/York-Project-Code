#!/bin/bash

extract_jorek_inputs(){
	./jorek2_postproc < $JOREK_TOOLS/quasi_linear_model/get_flux.pp
}

extract_jorek_mac_vars(){
	$JOREK_UTIL/extract_live_data.sh -si magnetic_energies > magnetic_energies_si.dat
	$JOREK_UTIL/extract_live_data.sh -si magnetic_growth_rates > magnetic_growth_rates_si.dat
}

extract_params(){
	extract_jorek_inputs
	extract_jorek_mac_vars

	#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
	#source $SCRIPT_DIR/../delta_psi_extraction/delta_psi_main.sh

	extract_delta_psi_all
}

model_params(){
	python3 -m experiments.jorek_growth_comparison.model_jorek_params -p -si -ex postproc/exprs_averaged_s00000.dat -q postproc/qprofile_s00000.dat
}


compare_ql_to_jorek() {
	python3 -m experiments.jorek_growth_comparison.compare_growth_rates -ql postproc/output/*jorek_model_m2_n1.zip -jg magnetic_growth_rates_si.dat -je magnetic_energies_si.dat -f postproc/exprs_four2d_s*_absolute_value_n001.dat -t postproc/times.txt -q postproc/qprofile_s00000.dat
}
