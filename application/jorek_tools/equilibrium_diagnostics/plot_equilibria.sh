#!/bin/bash

plot_pol_avg_profs() {
	postproc_output=$1
	# plq defined in macroscopic_vars_analysis.dat. Assume this is sourced
	plq -f "$postproc_output" -xi 1 -fs 4 3 -yi 3 -xl "r (m)" -yl "\$J_\phi\$ [JU]" -o j_profile.pdf
	plq -f "$postproc_output" -xi 1 -fs 4 3 -yi 7 -xl "r (m)" -yl "\$n\$ [JU]" -o n_profile.pdf
	plq -f "$postproc_output" -xi 1 -fs 4 3 -yi 8 -xl "r (m)" -yl "\$p\$ [JU]" -o p_profile.pdf
	plq -f "$postproc_output" -xi 1 -fs 4 3 -yi 9 -xl "r (m)" -yl "\$T\$ [JU]" -o t_profile.pdf
	plq -f "$postproc_output" -xi 1 -fs 4 3 -yi 10 -xl "r (m)" -yl "\$FF'\$ [JU]" -o ff_profile.pdf
}

