#!/bin/bash

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
	# Take pressure profile (col 6), multiply by mu_0 to convert to JOREK unit temperature
	# Assuming rho=1 over the profile. May need to change later.
	# Also, limit minimum temperature to 1e-8
        chease_col_filename=$1
        chease_profile $chease_col_filename "4*atan2(0,-1)*1e-7*\$6" | awk '{print $1 " " ($2 > 1e-8 ? $2 : 1e-8)}'
}
