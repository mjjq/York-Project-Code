#!/bin/bash

./prof < prof_namelist |& tee prof_output.out
cp EXPEQ_INIT EXPEQ
./chease < chease_namelist |& tee chease_output.out


echo "Generating profiles..."

./eqdsk2jorek < EQDSK_COCOS_02.OUT

./o.chease_to_cols chease_output.out chease_cols.out

#source $CHEASE_TOOLS/aliases.sh
#chease_temperature chease_cols.out > jorek_temperature
#chease_ffprime chease_cols.out > jorek_ffprime
#chease_cols_psin_convert chease_cols.out > chease_cols_psin.out
