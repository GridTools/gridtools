#!/bin/bash

export GTCMAKE_Boost_NO_BOOST_CMAKE=ON
export GTCMAKE_Boost_NO_SYSTEM_PATHS=ON
export GTCMAKE_GT_GCL_ONLY=OFF
export GTCMAKE_GT_ENABLE_PYUTILS=ON
export GTCMAKE_GT_TESTS_REQUIRE_FORTRAN_COMPILER=ON
export GTCMAKE_GT_TESTS_REQUIRE_C_COMPILER=ON
export GTCMAKE_CMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON
export GTCMAKE_TEST_USE_WRAPPERS_FOR_ALL_TESTS=OFF

export GTCMAKE_GT_ENABLE_BACKEND_X86=ON
export GTCMAKE_GT_ENABLE_BACKEND_MC=ON
export GTCMAKE_GT_ENABLE_BACKEND_NAIVE=ON

export GTRUN_SBATCH_TIME='00:15:00'
export GTRUN_SBATCH_EXCLUSIVE=''

export CTEST_PARALLEL_LEVEL=10

export OMP_PROC_BIND='true'
export OMP_WAIT_POLICY='active'
