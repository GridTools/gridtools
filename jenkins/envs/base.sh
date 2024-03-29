#!/bin/bash

export GTCMAKE_Boost_NO_BOOST_CMAKE=ON
export GTCMAKE_Boost_NO_SYSTEM_PATHS=ON
export GTCMAKE_GT_TESTS_REQUIRE_FORTRAN_COMPILER=ON
export GTCMAKE_GT_TESTS_REQUIRE_C_COMPILER=ON
export GTCMAKE_GT_TESTS_REQUIRE_OpenMP=ON
export GTCMAKE_GT_TESTS_REQUIRE_GPU=ON
export GTCMAKE_GT_TESTS_REQUIRE_Python=ON
export GT_ENABLE_STENCIL_DUMP=ON
export GTCMAKE_CMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON

export GTRUN_SBATCH_TIME='00:15:00'
export GTRUN_SBATCH_EXCLUSIVE=''

export CTEST_PARALLEL_LEVEL=10

export OMP_PROC_BIND='true'
export OMP_WAIT_POLICY='active'
