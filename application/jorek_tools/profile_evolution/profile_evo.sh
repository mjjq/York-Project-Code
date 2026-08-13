#!/bin/bash

function plot_midplane_profs() {
	plq -f postproc/exprs_midplane*.dat -yi 4 -xl "R (m)" -yl "\$n_e\$ (/m\$^3\$)" -fs 5.5 4 -o ne_mid.png "$@" &
	plq -f postproc/exprs_midplane*.dat -yi 5 -xl "R (m)" -yl "\$T_e\$ (eV)" -fs 5.5 4 -o T_e_mid.png "$@" &
	plq -f postproc/exprs_midplane*.dat -yi 6 -xl "R (m)" -yl "Pressure (Pa)" -fs 5.5 4 -o pressure_mid.png "$@" &
	plq -f postproc/exprs_midplane*.dat -yi 2 -xl "R (m)" -yl "\$J_\phi\$ (A/m\$^2\$)" -fs 5.5 4 -o jphi_mid.png "$@" &
}

function plot_psin_profs() {
	plq -f postproc/exprs_averaged*.dat -yi 5 -xl "\$\psi_N\$" -yl "\$n_e\$ (/m\$^3\$)" -fs 5.5 4 -o ne_psin.png "$@" &
	plq -f postproc/exprs_averaged*.dat -yi 6 -xl "\$\psi_N\$" -yl "\$T_e\$ (eV)" -fs 5.5 4 -o T_e_psin.png "$@" &
	plq -f postproc/exprs_averaged*.dat -yi 7 -xl "\$\psi_N\$" -yl "Pressure (Pa)" -fs 5.5 4 -o pressure_psin.png "$@" &
	plq -f postproc/exprs_averaged*.dat -yi 3 -xl "\$\psi_N\$" -yl "\$J_\phi\$ (A/m\$^2\$)" -fs 5.5 4 -o jphi_psin.png "$@" &
}
