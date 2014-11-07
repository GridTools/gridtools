#-*- coding: utf-8 -*-

from glob import glob
from os.path import join, basename, splitext
from setuptools import setup, Extension

include_dirs = ["/usr/local/include", 
                "/opt/local/include",
                "./src",
                "../communication_framework",
                "../hp2c_dycore",
                "../serialization_framework",
                "../shared_definitions",
                "../shared_infrastructure",
                "../stencil_framework",
                "../verification_framework"]
library_dirs = ["/usr/lib",
                "/usr/lib64",
                "/usr/local/lib",
                "/usr/local/lib64",
                "/opt/local/lib"]
libraries     = ["gomp"]            # OpenMP
extra_objects = [
                 "../../build/src/hp2c_dycore/libHP2CDycore.a",
                 "../../build/src/verification_framework/libVerificationFramework.a",
                 "../../build/src/serialization_framework/libSerializationFramework.a",
                 "../../build/src/shared_infrastructure/libSharedInfrastructure.a",
                 "../../build/src/communication_framework/libCommunicationFramework.a",
                 "../../build/libs/libjson/liblibjson.a",
                 "../../build/libs/gmock-gtest/libgmock-gtest.a"]

# Attempt to find libboost_python.so or some variant by searching through the
# library directories.
for d in library_dirs:
    libs = glob (join (d, "libboost_python3.so"))
    if not (libs):
        libs = glob (join (d, "libboost_python3*.so"))
    if libs:
        libname = basename(libs[0])         # basename
        libname = splitext(libname)[0]      # truncate postfix
        libname = libname[3:]               # truncate "lib"
        libraries.append (libname)
        break

# If we were unable to find the shared library go ahead in a default. It might
# be in an unofficial directory and an environment variable has been set that
# will point the compiler to it.
if len (libraries) < 2:
    libraries.append ('boost_python')

setup (
    name='gridtools4py',
    description="Python interface C++ library Gridtools",
    version="0.0.1",
    author="Lucas Benedicic",
    author_email="benedicic@cscs.ch",
    maintainer="Lucas Benedicic",
    maintainer_email="benedicic@cscs.ch",
    keywords="stencil jit cuda openmp",
    packages=["gridtools"],
    test_suite="tests",
    license="???",
    url="https://github.com/eth-cscs/gridtools",
    #ext_modules=[
    #    Extension(
    #        "stella._backend",
    #        sources=["src/boost/numpy/src/dtype.cpp",
    #                 "src/boost/numpy/src/matrix.cpp",
    #                 "src/boost/numpy/src/ndarray.cpp",
    #                 "src/boost/numpy/src/numpy.cpp",
    #                 "src/boost/numpy/src/scalars.cpp",
    #                 "src/boost/numpy/src/ufunc.cpp",
    #                 "src/_backend.cpp"],
    #        extra_compile_args=['-fopenmp'],        # OpenMP
    #        include_dirs=include_dirs,
    #        library_dirs=library_dirs,
    #        libraries=libraries,
    #        extra_objects=extra_objects
    #    )
    #]
)
