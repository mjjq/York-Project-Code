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
