#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

function extract_delta_psi_all()
{
    mkdir postproc
    $SCRIPT_DIR/time.sh log > postproc/times.txt
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