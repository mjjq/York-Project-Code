#!/bin/bash
source ${HOME}/.bashrc

rm -rf build
rm -rf install

make config cc=mpiicc prefix=${LIBS}/ParMETIS/install openmp=1 CFLAGS="-O2 -axCORE-AVX512 -D_POSIX_C_SOURCE=199309L" gklib_path=${LIBS}/GKlib/install metis_path=${LIBS}/METIS/install

make -j 16 install
