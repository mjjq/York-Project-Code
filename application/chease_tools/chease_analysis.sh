#!/bin/bash

cols(){
	cols_path=$1
	head -n 1 $cols_path | tr -s ' ' '\n' | tail -n +2 | awk '{print NR-1 " " $0}'
}

tp() {
	awk '{for(i=1;i<=NF;i++)a[i][NR]=$i}END{for(i in a)for(j in a[i])printf"%s"(j==NR?RS:FS),a[i][j]}' "${1+FS=$1}";
}

cols_row(){
	((head -n 1 $1 | sed 's/#//') && awk -v var="$2" '{if(NR==var) print $0}' $1) | tp
}

plot_time_avg_dr_profs() {
	plq ${@} -yi 1 -yerr 2 -xl "\$\psi_N\$" -yl "\$-s^2 D_R / \alpha\$" &
	plq ${@} -yi 3 -yerr 4 -xl "\$\psi_N\$" -yl "\$\alpha\$" &
	plq ${@} -yi 5 -yerr 6 -xl "\$\psi_N\$" -yl "\$s\$" &
}