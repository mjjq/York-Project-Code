#!/bin/bash

rm -rf build/

make config cc=mpiicc prefix=${LIBS}/METIS/install openmp=1 CFLAGS="-O2 -axCORE-AVX512 -D_POSIX_C_SOURCE=199309L" gklib_path=${LIBS}/GKlib/install i64=1 r64=1

#Note - had to add "link_directories(${GKLIB_PATH}/lib64)" to CMakeLists.txt as a place to look for libgklib.a
make -j 16 install
