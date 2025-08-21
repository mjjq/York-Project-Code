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
	cat useful_runs.txt | xargs find | grep chease_cols | xargs python3 -m chease_tools.dr_term_at_q -q 2.0
}

