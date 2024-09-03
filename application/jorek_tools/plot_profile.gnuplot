# set term x11 persist
set datafile commentschars "#"
set key noautotitle

# set format x "%.0s*10^{%T}"

set style line 1 \
	linecolor rgb '#0060ad' \
	linetype 1 linewidth 2 \
	pointtype 7 pointsize 1.5

header = system("head -n 1 ".filename)

num_columns = system("awk 'NR==3 {print NF}' ".filename)

set multiplot layout num_columns-1,1

do for [i=2:num_columns] {

	if (i==num_columns) {set xlabel word(header, 2);}

	set ylabel word(header, i+1)." [SI-units]"
	plot filename using 1:i with lines


}


# set xlabel '\$\\Psi_N\$'

unset multiplot

pause -1

