#!/bin/bash -l

module load CMake/3.5.2
module load GCC/5.3.0-2.26
#module load Boost/1.61.0-foss-2016a
module load Boost/1.58.0-Python-2.7.9
module load imkl/2017.0.098-iimpi-2017.00-GCC-5.4.0-2.26

#export BOOST_ROOT=/apps/all/Boost/1.61.0-foss-2016a/include
export BOOST_ROOT=/apps/all/Boost/1.58.0-Python-2.7.9/include
export GRIDTOOLS_ROOT=/home/kardoj/gridtools
export GTEST_ROOT=/home/kardoj/googletest/googletest

#set path to pardiso and MKL
#vim gridtools/examples/CMakeLists.txt

#cd gridtools/build
#cmake ..
#ccmake . -DGTEST_ROOT=/home/kardoj/googletest/googletest/build
#set these variables to ON
#USE_MPI
#USE_MPI_COMPILER
#CXX_11
