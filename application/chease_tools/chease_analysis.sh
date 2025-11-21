#!/bin/bash

list_chease_cols(){
	cols_path=$1
	head -n 1 $cols_path | tr -s ' ' '\n' | tail -n +2 | awk '{print NR-1 " " $0}'
}
