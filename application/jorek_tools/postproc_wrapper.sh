#!/bin/bash

postproc_time() {
	if [ $# -eq 0 ]
	then
		echo "USAGE: postproc_time <postproc_script> <t0> <t1 [optional]> <tstep [optional]>"
		return
	fi

	pp_script=$1
	t0=$2
	t1=$3
	tstep=$4

	if [ $# -eq 1 ]
	then
		./jorek2_postproc < "$pp_script"
		return
	fi

	if [ $# -eq 2 ]
	then
		sed "s/for step.*/for step $t0 do/" $pp_script > tmp.pp
	elif [ $# -eq 3 ]
	then
		sed "s/for step.*/for step $t0 to $t1 do/" $pp_script > tmp.pp
	elif [ $# -eq 4 ]
	then
		sed "s/for step.*/for step $t0 to $t1 by $tstep do/" $pp_script > tmp.pp
	fi


	./jorek2_postproc < tmp.pp
	rm tmp.pp
}
