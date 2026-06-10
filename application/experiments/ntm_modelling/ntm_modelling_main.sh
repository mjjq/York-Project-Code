#!/bin/bash

extract_mre_time_series() {
	python3 -m experiments.ntm_modelling.mre_time_series -c g_p*_t0*/chease_cols.out -w *w_measured.txt "$@"
}
