#!/bin/sh

mkdir -p ./build
rm -r ./build/*

make clean && make -j 32
make -j 32 jorek2_postproc
make -j 32 jorek2vtk
make -j 32 jorek2_poincare

mv jorek_model* ./build/jorek_model
mv jorek2_postproc jorek2vtk jorek2_poincare ./build

make clean && make -j 32 DEBUG=1
mv jorek_model* ./build/jorek_model_debug

./util/config.sh > ./build/settings.txt

git log -1 > ./build/git_info.txt
git branch >> ./build/git_info.txt
