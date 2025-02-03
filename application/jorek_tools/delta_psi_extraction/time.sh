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

logfile=$1

function extract_time_from_log() {
	grep -r "t_now" $logfile | sed 's/After step //' | sed 's/(t_now= //' | sed 's/)://'
}

function extract_mu0_rho0_from_log() {
	grep -r "mu0\*rho0" $logfile | awk 'NF>1{print $NF}'
}


function extract_time_to_si() {
	extract_time_from_log | awk -v t=$(extract_mu0_rho0_from_log) '{print $1 " " t*$2}'
}

extract_time_to_si
