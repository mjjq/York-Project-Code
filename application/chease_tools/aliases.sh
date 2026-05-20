#!/bin/bash

chease_cols_psin_convert() {
	chease_col_filename=$1
	awk 'NR == 1 { print $0; next } {$1 = $1^2; print}' $1 | sed 's/S-MESH/Psi_N/'
}

get_profile() {
	profile_name=$1
	sed -n "/$1/,/^\s*$/p" chease_output.out | sed -E 's/^[[:space:]]+//g' | sed -E 's/[[:space:]]+/\n/g'
}

chease_profile() {
        # Take pressure profile (col 6), multiply by mu_0 to convert to JOREK unit temperature
        # Assuming rho=1 over the profile. May need to change later.
        chease_col_filename=$1
	chease_col=$2
	awk_command="awk '{printf(\"%.10f %.12f\n\"), \$1^2, $2}'"
        cat $chease_col_filename | eval "$awk_command" | tail -n +2
}


chease_ffprime() {
	# FFprime is column 5
	# Escape the $ to avoid evaluation to parameter 5 in function
	chease_col_filename=$1
	chease_profile $chease_col_filename "\$5"
}

chease_temperature() {
	# Take pressure profile (col 6). The CHEASE output file is in terms
	# of CHEASE units. This implies CHEASE pressure is exactly equal
	# to JOREK temperature, assuming B0EXP=1 (see lab book).
	# Assuming rho=1 over the profile. May need to change later.
	# Also, limit minimum temperature to 1e-8
        chease_col_filename=$1
        chease_profile $chease_col_filename "\$6" | awk '{print $1 " " ($2 > 1e-8 ? $2 : $2)}'
}

chease_dr_at_q2() {
	python3 -m chease_tools.dr_term_at_q $@ -q 2.0
}

chease_tools_dir() {
	cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd
}

function chease_single_timeslice() {
        # Note: This requires rdcon and all relevant .in files in your cwd
        fullpath=$1
        filename=$(basename ${fullpath})
        mkdir $filename
        cd $filename
        cp ../* .
        cp $fullpath EXPEQ
        #sed -i "s/.*eq_filename.*/eq_filename=\"$filename\"/" equil.in
	echo "Running for $filename"
        ./chease < chease_namelist > chease_output.out
	./o.chease_to_cols chease_output.out chease_cols.out
	echo "Done for $filename"
        cd ..
}

function chease_parallel() {
        export -f chease_single_timeslice
        ncores=$1
        files="${@:2}"
        ls $files | xargs -t -P $ncores -I {} bash -c 'chease_single_timeslice "{}"'
}



source $(chease_tools_dir)/chease_analysis.sh
