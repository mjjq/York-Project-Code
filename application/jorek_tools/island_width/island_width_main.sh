#!/bin/bash

extract_calibrated_width_21() {
	python3 -m jorek_tools.island_width.calibrated_island_width -f postproc/exprs_four2d_*absolute*n001.dat -q postproc/qprofile_s00000.dat -t postproc/times.txt -w island_calibrations.txt "$@"
}

compare_delta_prime_jorek() {
	python3 -m experiments.ntm_modelling.ntm_modelling chease_cols.out -w w_measured.txt "$@"
}
