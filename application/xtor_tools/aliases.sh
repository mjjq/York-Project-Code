#!/bin/bash

load_xtor_mod_csd() {
	module purge

	module load rhel8/default-icl
	module load intel-oneapi-tbb
	module load intel-oneapi-mkl
	module load fftw
	module load petsc/3.17-icl
	module load netcdf-fortran
}

load_xtor_mod_pita() {
	module purge

	module load intel-oneapi-tbb
	module load intel-oneapi-mkl
	module load intel-oneapi-compilers
	module load fftw
	module load petsc
	module load netcdf-fortran
}

xtorpythonenv() {
	export PYTHONPATH=$PROJ_HOME/xtor_python
	source $PROJ_HOME/xtor_python/.venv/bin/activate
}

xtorgui() {
	python3 $HPC_SOFTWARE/xtor_python/gui.py
}

xtorprint() {
	python3 $PROJ_HOME/xtor_python/export_profiles.py
}
