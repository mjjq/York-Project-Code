#!/bin/bash
module purge
load_jorek_mod_csd

ScaLAPACKLIBS="-L${MKLROOT}/lib/intel64/ -lmkl_scalapack_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64"
export METIS_DIR="${LIBS}/METIS/install"
export ParMETIS_DIR="${LIBS}/ParMETIS/install"
export GKLIB_DIR="${LIBS}/GKlib/install"
echo ${ScaLAPACKLIBS}

rm -rf build
rm -rf install
mkdir build
mkdir install
cd build

cmake ../ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=../install \
	 -DCMAKE_CXX_COMPILER=mpiicpc \
        -DCMAKE_C_COMPILER=mpiicc \
        -DCMAKE_Fortran_COMPILER=mpiifort \
        -DSTRUMPACK_USE_CUDA=OFF \
        -DSTRUMPACK_USE_MPI=ON \
        -DSTRUMPACK_USE_OPENMP=ON \
        -DTPL_ENABLE_PARMETIS=ON \
        -DTPL_SCALAPACK_LIBRARIES="$ScaLAPACKLIBS" \
        -DTPL_ENABLE_BPACK=OFF \
        -DTPL_ENABLE_ZFP=OFF \
        -DTPL_ENABLE_SLATE=OFF \
        -DTPL_ENABLE_PARMETIS=ON \
	-DTPL_ENABLE_SCOTCH=OFF \
	-DTPL_METIS_INCLUDE_DIR="${METIS_DIR}/include ${GKLIB_DIR}/include" \
        -DTPL_METIS_LIBRARIES="-L${METIS_DIR}/lib -lmetis -L${GKLIB_DIR}/lib64 -lGKlib" \
	-DTPL_PARMETIS_INCLUDE_DIR="${ParMETIS_DIR}/include" \
        -DTPL_PARMETIS_LIBRARIES="-L${PARMETIS_DIR}/lib -lparmetis" \

make -j4
make install
make examples -j4
