#!/bin/sh

mkdir -p ./build
rm -r ./build/*

make clean && make -j 32
make jorek2_postproc
make jorek2vtk

mv jorek_model* ./build/jorek_model
mv jorek2_postproc jorek2vtk ./build

make clean && make -j 32 DEBUG=1
mv jorek_model* ./build/jorek_model_debug

./util/config.sh > ./build/settings.txt
