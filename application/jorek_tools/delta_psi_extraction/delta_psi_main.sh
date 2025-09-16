#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

function find_unprocessed_h5()
{
	ls jorek[0-9]*.h5 | sed 's/jorek//; s/\.h5//' > all.tmp
	ls postproc/exprs_four2d_s*_absolute*.dat | grep -o 's[0-9]\+' | sed 's/s//' | sort -u > processed.tmp

	#  -2     suppress column 2 (lines unique to FILE2)
        #  -3     suppress column 3 (lines that appear in both files)
	comm -23 all.tmp processed.tmp | tr '\n' ','

	rm all.tmp processed.tmp
}

function extract_delta_psi_all()
{
    mkdir postproc

    source $SCRIPT_DIR/time_restart.sh
    get_time_map > postproc/times.txt

    ./jorek2_postproc < $SCRIPT_DIR/qprofile.pp
    ./jorek2_postproc < $SCRIPT_DIR/fourier_r_minor.pp
}

function plot_delta_psi_time()
{
    python3 -m jorek_tools.delta_psi_extraction.plot_delta_psi_vs_time -f postproc/exprs_four2d_s0*abs*n001.dat -q postproc/qprofile_s00000.dat -t postproc/times.txt
}

function plot_delta_psi_psin()
{
    python3 -m jorek_tools.delta_psi_extraction.plot_delta_psi_vs_psin -f $1 -t postproc/times.txt
}
