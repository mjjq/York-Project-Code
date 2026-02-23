# Building dependencies

IMPORTANT!! Make sure to hide any compiled Scotch/PtScotch libraries. 
These contain metis.h header files which conflict with the version of 
METIS we'll be compiling below, and will create compiler errors!

## GKLib

`git clone https://github.com/KarypisLab/GKlib.git`

`cd GKlib`

`make config`

`make`

## METIS

`git clone https://github.com/KarypisLab/METIS.git`

Open `CMakeLists.txt` and add `link_directories(${GKLIB_PATH}/lib64)` line

Then, `source $JOREK_TOOLS/build/make_metis.sh`

## ParMETIS

`git clone https://github.com/KarypisLab/ParMETIS.git`

Open `CMakeLists.txt` and add `link_directories(${GKLIB_PATH}/lib64)` line

Then, `source $JOREK_TOOLS/build/make_parmetis.sh`

## StrumPACK

Download the latest release of StrumPACK

`wget https://github.com/pghysels/STRUMPACK/archive/v7.1.0.tar.gz`

`tar -zxvf v7.1.0.tar.gz`

`cd` to the STRUMPACK directory then run

`source $JOREK_TOOLS/build/make_strumpack.sh`


# JOREK

Copy the STRUMPACK Makefile.inc from Make.inc/ into the root of the JOREK
directory

Then run `make -j 32`
