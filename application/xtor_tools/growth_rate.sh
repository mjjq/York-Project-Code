#!/bin/bash

max_growth_rate_single(){
	logfile=$1
	grep -A 1 "taux de croissance n =            1" $logfile | grep magnetic | awk '{print $2}' | sort -rg | head -n 1
}

get_mks_time(){
	logfile=$1
	grep "t_to_mks" $logfile | awk '{print $3}'
}

max_growth_rate(){
	for logfile in "$@"; do
		mks_time=$(get_mks_time $logfile)
		max_growth_rate_single $logfile | awk -v ta=$mks_time '{print $1/ta/sqrt(2)}'
	done
}
