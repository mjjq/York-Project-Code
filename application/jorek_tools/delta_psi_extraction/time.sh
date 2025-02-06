#!/bin/bash

function usage() {
	echo ""
	echo "Extract timestep vs SI unit time"
	echo ""
	echo "Usage: `basename $0` <path to logfile>"
	echo ""
}

if [ $# -lt 1 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
	usage
	exit;
fi

logfiles=$@

function extract_time_from_log() {
	grep -r "t_now" $1 | sed 's/After step //' | sed 's/(t_now= //' | sed 's/)://'
}

function extract_abs_step_from_log() {
	grep -E "time step" $1 | awk '{ printf("%05d %05d\n", $6, $7) }'
}

function abs_time_from_log() {
	join <(extract_abs_step_from_log $1) <(extract_time_from_log $1) | awk '{ print $2 " " $3 }'
}

function extract_mu0_rho0_from_log() {
	grep -r "mu0\*rho0" $1 | awk 'NF>1{print $NF}'
}


function extract_time_to_si() {
	abs_time_from_log $1 | awk -v t=$(extract_mu0_rho0_from_log $1) '{print $1 " " t*$2}'
}

for logfile in $logfiles
do
	#echo $logfile
	extract_time_to_si $logfile
done
