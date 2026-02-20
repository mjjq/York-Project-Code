#!/bin/bash

function extract_all_profiles() {
	p_folder=$JOREK_TOOLS/profile_evolution
	./jorek2_postproc < $p_folder/get_midplane.pp
	./jorek2_postproc < $p_folder/qprofile.pp
	./jorek2_postproc < $p_folder/get_profiles.pp
}

function extract_all_profiles_hi() {
        p_folder=$JOREK_TOOLS/profile_evolution
        ./jorek2_postproc < $p_folder/get_midplane_hi.pp
        ./jorek2_postproc < $p_folder/qprofile_hi.pp
        ./jorek2_postproc < $p_folder/get_profiles_hi.pp
}

